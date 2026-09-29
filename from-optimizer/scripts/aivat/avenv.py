import json, math, os, sys
import numpy as np
root = os.environ['AVROOT']
sys.path.insert(0, root)
from negpluribus import fast
from negpluribus.abstraction import load_bucketer
from negpluribus.cfr.game import GameSpec
from negpluribus.engine import Street
from negpluribus.eval.aivat import hand_from_duel
from negpluribus.eval.aivat_fast import FastAivat, make_game, root_table, hand_to_dict
from negpluribus.fast.tables import table_name
from negpluribus.fast.blueprint import load_blueprint
from negpluribus.fast.trainer import core_bucketer
S = "/tmp/claude-0/-home-user-Zootop/780badec-1fee-5bac-85c5-e13d71ebb617/scratchpad"; B = "/home/user/duel200"
core = fast.core()
bk = load_bucketer(f"{B}/buckets_base_s0.json")
spec = GameSpec(n_players=2, stack_bb=200, max_street=Street.RIVER, preflop_fracs=(0.5, 1.0, 3.0), postflop_fracs=(0.5, 1.0, 2.0, 4.0),
                max_raises_per_street=3, n_buckets=bk.n_buckets, bucket_kind=getattr(bk, "kind", "ehs"))
cbk = core_bucketer(bk)
tables = core.BucketTables(); tables.load(os.path.join("/home/user/data/bucket_tables_pot16", table_name(cbk)), cbk)
tb = core.TabulatedBucketer(cbk, tables)
bp = load_blueprint(f"{B}/blueprint_base_s0.bin", backend="cpp", n_players=2)
game = make_game(spec, tb, bp)
rt = root_table(game, 256, seed=0, threads=4, cache_path=f"{S}/root_base.npz",
                identity={"blueprint": "blueprint_base_s0.bin", "buckets": "buckets_base_s0.json", "grid": ["0.5,1.0,3.0", "0.5,1.0,2.0,4.0", 3, 200]})
recs = [json.loads(l) for l in open(os.environ.get('AVLOG', '/tmp/claude-0/-home-user-Zootop/780badec-1fee-5bac-85c5-e13d71ebb617/scratchpad/hands2000.jsonl'))][: int(os.environ.get("AVN", "2000"))]
hands = [hand_to_dict(hand_from_duel(r, stack=20000, hand_id=i, known="hero")) for i, r in enumerate(recs)]
deals = [int(r['deal']) for r in recs]
