#include "cfr_engine.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>

void print_usage() {
    std::cerr << "Usage: train_cfr [options]\n"
              << "  --iterations N   Number of CFR iterations (default: 5000)\n"
              << "  --output PATH    Binary output for strategy (default: strategy.bin)\n"
              << "  --checkpoint PATH Checkpoint file for resume (default: checkpoint.bin)\n"
              << "  --resume         Resume from checkpoint\n"
              << "  --mc-sims N      MC sims per discard candidate (default: 60)\n";
}

int main(int argc, char** argv) {
    int iterations = 5000;
    const char* output = "strategy.bin";
    const char* checkpoint = "checkpoint.bin";
    bool resume = false;
    int validate_every = 0;  // 0 = no intermediate saves

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--iterations") == 0 && i+1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--output") == 0 && i+1 < argc) output = argv[++i];
        else if (std::strcmp(argv[i], "--checkpoint") == 0 && i+1 < argc) checkpoint = argv[++i];
        else if (std::strcmp(argv[i], "--resume") == 0) resume = true;
        else if (std::strcmp(argv[i], "--validate-every") == 0 && i+1 < argc) validate_every = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--help") == 0) { print_usage(); return 0; }
    }

    CFRTrainer trainer;

    if (resume) {
        trainer.load_checkpoint(checkpoint);
        std::cout << "Resumed from checkpoint: " << trainer.iterations << " iters, "
                  << trainer.nodes.size() << " nodes" << std::endl;
    }

    std::cout << "Training " << iterations << " iterations..." << std::endl;

    auto t0 = std::chrono::steady_clock::now();
    int print_every = std::max(1, iterations / 20);

    for (int i = 0; i < iterations; i++) {
        trainer.train_one();

        if ((i + 1) % print_every == 0 || i == iterations - 1) {
            auto t1 = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            double ips = (i + 1) / elapsed;
            std::cout << "  iter " << (i+1) << "/" << iterations
                      << "  " << elapsed << "s  "
                      << trainer.nodes.size() << " nodes  "
                      << ips << " it/s" << std::endl;
        }

        // Periodic save for validation
        if (validate_every > 0 && (i + 1) % validate_every == 0) {
            std::cout << "\n--- Saving at iter " << trainer.iterations
                      << " (" << trainer.nodes.size() << " nodes) ---" << std::endl;
            trainer.save_binary(output);
            trainer.save_checkpoint(checkpoint);
            // Touch a signal file so the validation script knows to run
            std::string signal = std::string(output) + ".ready";
            std::ofstream(signal) << trainer.iterations << std::endl;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Done: " << trainer.iterations << " total iters in "
              << elapsed << "s, " << trainer.nodes.size() << " nodes" << std::endl;

    trainer.save_binary(output);
    std::cout << "Saved strategy to " << output << std::endl;

    trainer.save_checkpoint(checkpoint);
    std::cout << "Saved checkpoint to " << checkpoint << std::endl;

    return 0;
}
