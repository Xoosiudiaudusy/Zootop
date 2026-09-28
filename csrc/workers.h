// Error handling of the trainers' worker threads (no behaviour change when nothing fails).
//
// An exception in a worker thread that escapes it ends the process (std::terminate; 0xC0000409 on Windows), and so
// does a joinable std::thread destroyed when starting the pool fails half way.  run_workers() runs work(tid) for
// tid = 0 .. T-1 (0 on the caller), keeps the first error, calls after(tid) for every slot whether its work
// failed or its thread never started (the table group's leave()), joins every started thread, and leaves the
// error in `errs` for the caller to raise once the tables are in a known state.
#pragma once

#include <atomic>
#include <exception>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace negp {

struct WorkerErrors {
    std::atomic<bool> failed{false};  // loops poll it to stop early
    void record(const std::string& what) {
        std::lock_guard<std::mutex> lk(mu_);
        if (msg_.empty()) msg_ = what;
        failed.store(true, std::memory_order_release);
    }
    std::string message() const {
        std::lock_guard<std::mutex> lk(mu_);
        return msg_;
    }
    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        msg_.clear();
        failed.store(false);
    }

private:
    mutable std::mutex mu_;
    std::string msg_;
};

// the message of the exception being handled (std::bad_alloc: "out of memory")
inline std::string current_error_text() {
    try {
        throw;
    } catch (const std::bad_alloc&) {
        return "out of memory";
    } catch (const std::exception& e) {
        return e.what();
    } catch (...) {
        return "unknown error";
    }
}

template <class W, class A>
void run_workers(int T, WorkerErrors& errs, W&& work, A&& after) {
    auto guarded = [&](int tid) {
        try {
            work(tid);
        } catch (...) {
            errs.record(current_error_text());
        }
        try {
            after(tid);
        } catch (...) {
            errs.record(current_error_text());
        }
    };
    if (T <= 1) {
        guarded(0);
        return;
    }
    std::vector<std::thread> pool;
    try {
        pool.reserve((size_t)T - 1);
        for (int t = 1; t < T; t++) pool.emplace_back(guarded, t);
    } catch (...) {
        errs.record("starting the worker threads: " + current_error_text());
        for (int t = (int)pool.size() + 1; t < T; t++) {
            try {
                after(t);  // the slots that never ran still leave (a table group waits for them otherwise)
            } catch (...) {
            }
        }
    }
    guarded(0);
    for (auto& th : pool) th.join();
}

}  // namespace negp
