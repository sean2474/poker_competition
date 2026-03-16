/*
 * Full CFR Trainer — enumerate all hand pairs per board.
 * 
 * Unlike MCCFR which samples 1 deal per iteration, this enumerates
 * ALL possible (hero_hand, opp_hand) pairs for each board configuration.
 *
 * Structure:
 *   1. Preflop: same as MCCFR (canonical 5-card, too many combos for full enum)
 *   2. Post-discard: for each sampled board, enumerate all hand pairs
 *      and traverse the full game tree, updating all strategies simultaneously.
 *
 * Each "iteration" = 1 sampled (5-card deal × 2 + board) → full hand pair enumeration
 * This is ~120×105 = 12,600 traversals per iteration vs MCCFR's 1.
 *
 * Usage:
 *   ./full_cfr_train --iterations 1000 --threads 32 --output strategy.bin
 */

#include "cfr_engine.h"
#include <iostream>
#include <chrono>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>

void print_usage() {
    std::cout << "Full CFR Trainer\n"
              << "  --iterations N   Number of board samples (default: 1000)\n"
              << "  --threads N      Number of threads (default: 1)\n"
              << "  --output PATH    Strategy binary output\n"
              << "  --checkpoint PATH Checkpoint file for resume\n"
              << "  --resume         Resume from checkpoint\n";
}

int main(int argc, char** argv) {
    int iterations = 1000;
    const char* output = "full_strategy.bin";
    const char* checkpoint = "full_checkpoint.bin";
    bool resume = false;
    int num_threads = 1;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--iterations") == 0 && i+1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--output") == 0 && i+1 < argc) output = argv[++i];
        else if (std::strcmp(argv[i], "--checkpoint") == 0 && i+1 < argc) checkpoint = argv[++i];
        else if (std::strcmp(argv[i], "--resume") == 0) resume = true;
        else if (std::strcmp(argv[i], "--threads") == 0 && i+1 < argc) num_threads = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--help") == 0) { print_usage(); return 0; }
    }

    CFRTrainer trainer;

    if (resume) {
        trainer.load_checkpoint(checkpoint);
        std::cout << "Resumed from checkpoint: " << trainer.iterations << " iters, "
                  << trainer.nodes.size() << " nodes" << std::endl;
    }

    std::cout << "Full CFR Training: " << iterations << " board samples"
              << " (" << num_threads << " threads)" << std::endl;

    auto t0 = std::chrono::steady_clock::now();
    std::atomic<int> completed{0};
    std::atomic<bool> done{false};

    // Monitor thread
    std::thread monitor([&]() {
        while (!done) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            int cur = completed.load();
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t0).count();
            double ips = (elapsed > 0) ? cur / elapsed : 0;
            double pct = 100.0 * cur / iterations;
            double eta = (ips > 0) ? (iterations - cur) / ips : 0;
            int eta_m = (int)(eta / 60);
            int eta_s = (int)(eta) % 60;
            int bar_width = 30;
            int filled = (int)(bar_width * cur / iterations);
            std::string bar(filled, '#');
            bar += std::string(bar_width - filled, '-');
            printf("\r  [%s] %5.1f%%  %d/%d  %.1f boards/s  %dk nodes  ETA %dm%02ds   ",
                   bar.c_str(), pct, cur, iterations, ips,
                   (int)(trainer.nodes.size() / 1000), eta_m, eta_s);
            fflush(stdout);
        }
        printf("\n");
    });

    // Worker function: sample a board, enumerate ALL hero × opp hand pairs
    auto worker = [&](int n_iters) {
        for (int iter = 0; iter < n_iters; iter++) {
            // Sample random board (5 community cards)
            int deck[DECK_SIZE];
            shuffle_deck(deck);
            int community[5];
            std::copy(deck, deck + 5, community);
            
            // All cards not on the board
            bool on_board[DECK_SIZE] = {};
            for (int j = 0; j < 5; j++) on_board[community[j]] = true;
            int pool[DECK_SIZE]; int np = 0;
            for (int c = 0; c < DECK_SIZE; c++) if (!on_board[c]) pool[np++] = c;
            
            // Enumerate ALL (hero_2cards, opp_2cards) pairs from pool
            // C(22,2) = 231 hero hands × C(20,2) = 190 opp hands per hero = ~44k pairs
            // But hero and opp can't share cards: ~231 × ~190 / overlap ≈ ~35k valid pairs
            
            for (int hi = 0; hi < np; hi++) {
                for (int hj = hi + 1; hj < np; hj++) {
                    int p0_hand[2] = {pool[hi], pool[hj]};
                    // Fake 5-card hand (preflop key uses this, but post-discard doesn't need exact 5)
                    int p0_hand5[5] = {pool[hi], pool[hj], -1, -1, -1};
                    int p0_disc[3] = {-1, -1, -1};
                    
                    for (int oi = 0; oi < np; oi++) {
                        if (oi == hi || oi == hj) continue;
                        for (int oj = oi + 1; oj < np; oj++) {
                            if (oj == hi || oj == hj) continue;
                            
                            int p1_hand[2] = {pool[oi], pool[oj]};
                            int p1_hand5[5] = {pool[oi], pool[oj], -1, -1, -1};
                            int p1_disc[3] = {-1, -1, -1};
                            
                            GameState state;
                            // Start from street 1 (flop) since we're post-discard
                            state.street = 1;
                            state.bets[0] = 2; state.bets[1] = 2; // after blinds equalized
                            state.current_player = 0;
                            state.min_raise = BIG_BLIND;
                            
                            trainer.cfr(state, p0_hand, p1_hand, p0_hand5, p1_hand5,
                                       community, p0_disc, p1_disc, 1.0, 1.0);
                        }
                    }
                }
            }
            
            trainer.iterations.fetch_add(1, std::memory_order_relaxed);
            completed.fetch_add(1, std::memory_order_relaxed);
        }
    };

    // Launch threads
    std::vector<std::thread> threads;
    int per_thread = iterations / num_threads;
    int extra = iterations % num_threads;
    for (int t = 0; t < num_threads; t++) {
        int count = per_thread + (t < extra ? 1 : 0);
        threads.emplace_back(worker, count);
    }
    for (auto& t : threads) t.join();
    done = true;
    monitor.join();

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Done: " << trainer.iterations << " iters in "
              << elapsed << "s, " << trainer.nodes.size() << " nodes" << std::endl;

    trainer.save_binary(output);
    std::cout << "Saved strategy to " << output << std::endl;

    trainer.save_checkpoint(checkpoint);
    std::cout << "Saved checkpoint to " << checkpoint << std::endl;

    return 0;
}
