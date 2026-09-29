// Python bindings of csrc/aivat.h (included by bindings.cpp just before the module definition, so the
// helpers above it - spec_from_dict, to_cards - are in scope).  negpluribus/eval/aivat_fast.py wraps them.
#pragma once
#include "aivat.h"

static negp::aiv::HandIn aivat_hand_from_py(const py::dict& d) {
    negp::aiv::HandIn h;
    h.hand_id = d["hand_id"].cast<int>();
    const std::vector<int> stacks = d["stacks"].cast<std::vector<int>>();
    if (stacks.size() != 2) throw std::invalid_argument("aivat: two stacks");
    h.stacks[0] = stacks[0];
    h.stacks[1] = stacks[1];
    h.button = d["button"].cast<int>();
    h.sb = d["sb"].cast<int>();
    h.bb = d["bb"].cast<int>();
    h.known_seat = d["known_seat"].cast<int>();
    const py::sequence holes = d["holes"].cast<py::sequence>();
    for (int s = 0; s < 2; s++) {
        const std::vector<int> hs = to_cards(holes[(size_t)s].cast<py::sequence>());
        if (hs.size() != 2) throw std::invalid_argument("aivat: two hole cards per seat");
        h.holes[s][0] = hs[0];
        h.holes[s][1] = hs[1];
    }
    h.board = to_cards(d["board"].cast<py::sequence>());
    for (auto a : d["actions"].cast<py::sequence>()) {
        const py::sequence p = a.cast<py::sequence>();
        h.actions.emplace_back(p[0].cast<int>(), p[1].cast<int>());
    }
    if (d.contains("x_rows") && !d["x_rows"].is_none()) {
        for (auto e : d["x_rows"].cast<py::sequence>()) {
            negp::aiv::LoggedRowsIn r;
            if (!e.is_none()) {
                const py::dict ed = e.cast<py::dict>();
                r.present = true;
                for (auto a : ed["actions"].cast<py::sequence>()) {
                    const py::sequence p = a.cast<py::sequence>();
                    r.actions.emplace_back(p[0].cast<int>(), p[1].cast<int>());
                }
                const std::string raw = ed["q"].cast<std::string>();  // little-endian uint16
                if (raw.size() % 2) throw std::invalid_argument("aivat: logged rows must be uint16");
                r.q.resize(raw.size() / 2);
                for (size_t i = 0; i < r.q.size(); i++)
                    r.q[i] = (uint16_t)((unsigned char)raw[2 * i] | ((unsigned)(unsigned char)raw[2 * i + 1] << 8));
            }
            h.x_rows.push_back(std::move(r));
        }
    }
    return h;
}

static py::dict aivat_out_to_py(const negp::aiv::HandOut& o) {
    py::dict d;
    d["hand_id"] = o.hand_id;
    d["net"] = o.net;
    d["base"] = o.base;
    d["value"] = o.value;
    py::list terms;
    for (const auto& t : o.terms) terms.append(py::make_tuple(t.kind, t.street, t.k, t.value));
    d["terms"] = terms;
    py::list trace;
    for (const auto& t : o.trace) trace.append(py::make_tuple(t.first, t.second));
    d["trace"] = trace;
    d["rollouts"] = o.rollouts;
    d["rollout_steps"] = o.rollout_steps;
    d["river_trees"] = o.river_trees;
    d["seconds"] = o.seconds;
    return d;
}

static void register_aivat(py::module_& m) {
    using namespace negp::aiv;
    m.def("aivat_stream_seed", [](const std::vector<uint64_t>& parts) {
        uint64_t h = 0x6A09E667F3BCC908ULL;
        for (uint64_t p : parts) h = mix64((h ^ p) + GOLD);
        return h;
    }, "tests: aivat.stream_seed");
    m.def("aivat_uniforms", [](uint64_t seed, int n) {
        CounterRng r(seed);
        std::vector<double> out((size_t)n);
        for (double& u : out) u = r.uniform();
        return out;
    }, "tests: the first n uniforms of a CounterRng");
    m.def("aivat_sampling_law", [](const std::vector<double>& p) {
        std::vector<double> out(p.size());
        sampling_law(p.data(), (int)p.size(), out.data());
        return out;
    });
    m.def("aivat_coin_prob", &coin_prob);
    m.def("aivat_set_lanes", [](int n) { Evaluator::set_lanes(n); return Evaluator::lanes_in_flight(); }, py::arg("n"),
          "rollouts in flight per branch (1..8; 1 = one after another, the default); the numbers do not depend on it");
    m.def("aivat_pair_orbits", []() {
        int n = 0;
        std::vector<int32_t> o;
        {
            py::gil_scoped_release nogil;
            o = pair_orbits(n);
        }
        return py::make_tuple(n, py::bytes(reinterpret_cast<const char*>(o.data()), o.size() * sizeof(int32_t)));
    }, "(number of classes, int32 bytes of the 1326 x 1326 class ids of ordered hole pairs (-1: overlapping))");
    m.def("aivat_build_tables", [](std::shared_ptr<negp::Bucketer> bk, int threads, const std::vector<int>& streets) {
        for (int s : streets)
            if (s < negp::FLOP || s > negp::RIVER) throw std::invalid_argument("streets: 1 (flop), 2 (turn), 3 (river)");
        auto t = std::make_shared<negp::BucketTables>();
        py::dict info;
        {
            py::gil_scoped_release nogil;
            for (int s : streets) {
                const auto t0 = std::chrono::steady_clock::now();
                uint64_t misses = 0;
                if (s == negp::RIVER) {
                    if (!build_river_table_fast(*bk, *t, threads)) misses = build_table_through_bucketer(*bk, *t, s, threads);
                } else {
                    misses = build_table_through_bucketer(*bk, *t, s, threads);
                }
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                py::gil_scoped_acquire g;
                info[py::int_(s)] = py::make_tuple(misses, sec);
            }
        }
        return py::make_tuple(t, info);
    }, py::arg("bucketer"), py::arg("threads") = 1, py::arg("streets") = std::vector<int>{1, 2, 3},
       "bucket tables of a fitted core bucketer: the river from per-board batches, flop/turn through bucket() "
       "(load a warm bucket cache first); returns (BucketTables, {street: (computed values, seconds)})");

    py::class_<Game, std::shared_ptr<Game>>(m, "AivatGame")
        .def(py::init([](const py::dict& spec, std::shared_ptr<negp::Bucketer> bk, std::shared_ptr<negp::BlueprintTable> bp) {
            return std::make_shared<Game>(spec_from_dict(spec), std::move(bk), std::shared_ptr<const negp::BlueprintTable>(bp));
        }), py::arg("spec"), py::arg("bucketer"), py::arg("blueprint"));

    py::class_<PreflopEquity, std::shared_ptr<PreflopEquity>>(m, "AivatPreflopEquity")
        .def_static("build", [](int threads, const py::object& only) {
            std::vector<int> cls;
            const bool part = !only.is_none();
            if (part) cls = only.cast<std::vector<int>>();
            py::gil_scoped_release nogil;
            return PreflopEquity::build(threads, part ? &cls : nullptr);
        }, py::arg("threads") = 1, py::arg("only") = py::none(),
           "exact heads-up preflop all-in equity per suit class of the hole pair: wins - losses over the 1,712,304 boards")
        .def_static("from_net", [](const std::vector<int32_t>& net) {
            auto t = std::make_shared<PreflopEquity>();
            t->orbit = pair_orbits(t->n_classes);
            if ((int)net.size() != t->n_classes) throw std::invalid_argument("preflop equity: one value per pair class");
            t->net = net;
            t->have.assign(net.size(), 1);
            return t;
        })
        .def("class_of", [](const PreflopEquity& t, int c, int d) { return t.orbit[(size_t)c * NC + (size_t)d]; })
        .def_property_readonly("n_classes", [](const PreflopEquity& t) { return t.n_classes; })
        .def_property_readonly("net", [](const PreflopEquity& t) { return t.net; })
        .def_readonly_static("boards", &PREFLOP_BOARDS);

    py::class_<RootTable, std::shared_ptr<RootTable>>(m, "AivatRootTable")
        .def_static("build", [](const Game& g, int rollouts, uint64_t seed, int threads) {
            py::gil_scoped_release nogil;
            return Evaluator::build_root_table(g, rollouts, seed, threads);
        }, py::arg("game"), py::arg("rollouts"), py::arg("seed") = 0, py::arg("threads") = 1)
        .def_static("from_values", [](const std::vector<double>& v0, const std::vector<double>& v1) {
            auto rt = std::make_shared<RootTable>();
            rt->orbit = pair_orbits(rt->n_classes);
            if ((int)v0.size() != rt->n_classes || (int)v1.size() != rt->n_classes)
                throw std::invalid_argument("root table: one value per class and seat (" + std::to_string(rt->n_classes) + ")");
            rt->values[0] = v0;
            rt->values[1] = v1;
            rt->finish();
            return rt;
        })
        .def_property_readonly("n_classes", [](const RootTable& r) { return r.n_classes; })
        .def_property_readonly("values", [](const RootTable& r) { return py::make_tuple(r.values[0], r.values[1]); })
        .def_property_readonly("mean", [](const RootTable& r) { return py::make_tuple(r.mean[0], r.mean[1]); });

    py::class_<Evaluator, std::shared_ptr<Evaluator>>(m, "AivatEvaluator")
        .def(py::init([](std::shared_ptr<Game> g, const std::vector<int>& rollouts, int eq_samples, uint64_t seed, const py::object& root,
                         const py::object& preflop) {
            ValueParams vp;
            if (rollouts.size() > 4) throw std::invalid_argument("rollouts: at most one per street (preflop, flop, turn)");
            for (size_t i = 0; i < rollouts.size(); i++) vp.rollouts[i] = rollouts[i];
            vp.eq_samples = eq_samples;
            vp.seed = seed;
            std::shared_ptr<const RootTable> rt;
            if (!root.is_none()) rt = root.cast<std::shared_ptr<RootTable>>();
            std::shared_ptr<const PreflopEquity> pf;
            if (!preflop.is_none()) pf = preflop.cast<std::shared_ptr<PreflopEquity>>();
            return std::make_shared<Evaluator>(std::shared_ptr<const Game>(g), vp, rt, pf);
        }), py::arg("game"), py::arg("rollouts"), py::arg("eq_samples") = 2000, py::arg("seed") = 0, py::arg("root") = py::none(),
            py::arg("preflop") = py::none())
        .def("evaluate", [](const Evaluator& e, const py::dict& hand, bool trace) {
            const HandIn h = aivat_hand_from_py(hand);
            HandOut o;
            {
                py::gil_scoped_release nogil;
                o = e.evaluate(h, trace);
            }
            return aivat_out_to_py(o);
        }, py::arg("hand"), py::arg("trace") = false)
        .def("evaluate_many", [](const Evaluator& e, const py::list& hands, int threads) {
            std::vector<HandIn> hs;
            hs.reserve(py::len(hands));
            for (auto h : hands) hs.push_back(aivat_hand_from_py(h.cast<py::dict>()));
            std::vector<HandOut> out;
            {
                py::gil_scoped_release nogil;
                out = e.evaluate_many(hs, threads);
            }
            py::list res;
            for (const HandOut& o : out) res.append(aivat_out_to_py(o));
            return res;
        }, py::arg("hands"), py::arg("threads") = 1)
        .def("root_value", [](const Evaluator& e, int pos, int ci, int di, int cls, int rollouts) {
            return e.root_value(pos, ci, di, cls, rollouts);
        }, "tests: one root-table entry computed directly")
        .def("branch_values", [](const Evaluator& e, const py::dict& hand, int k, const py::object& action, const std::vector<int>& fixed, int node,
                                 const std::vector<int>& combos_needed) {
            const HandIn h = aivat_hand_from_py(hand);
            std::vector<std::pair<int, int>> ov;
            if (!action.is_none()) {
                const py::sequence a = action.cast<py::sequence>();
                ov.emplace_back(a[0].cast<int>(), a[1].cast<int>());
            }
            std::vector<double> out;
            {
                py::gil_scoped_release nogil;
                out = e.branch_values(h, k, ov, fixed, node, combos_needed);
            }
            return out;
        }, "tests: the heuristic's values of one branch (actions[:k] replayed, then `action` or actions[k], next cards `fixed`)");
}
