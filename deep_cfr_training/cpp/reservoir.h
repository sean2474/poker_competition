#pragma once
#include <vector>
#include <random>
#include <array>
#include <mutex>
#include "game_state.h"

struct Sample {
    float features[FEATURE_DIM];
    float values[NUM_ACTIONS];
    float valid_mask[NUM_ACTIONS];
    int iteration;
};

class ReservoirBuffer {
public:
    int capacity;
    std::vector<Sample> buffer;
    int count = 0;
    std::mutex mtx;

    ReservoirBuffer(int cap = 2000000) : capacity(cap) {
        buffer.reserve(std::min(cap, 100000));
    }

    void add(const float* features, const float* values, const float* mask, int iter, std::mt19937& rng) {
        Sample s;
        std::memcpy(s.features, features, FEATURE_DIM * sizeof(float));
        std::memcpy(s.values, values, NUM_ACTIONS * sizeof(float));
        std::memcpy(s.valid_mask, mask, NUM_ACTIONS * sizeof(float));
        s.iteration = iter;

        std::lock_guard<std::mutex> lock(mtx);
        count++;
        if ((int)buffer.size() < capacity) {
            buffer.push_back(s);
        } else {
            std::uniform_int_distribution<int> dist(0, count - 1);
            int idx = dist(rng);
            if (idx < capacity) {
                buffer[idx] = s;
            }
        }
    }

    int size() const { return (int)buffer.size(); }
};
