CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

all: test_cards test_abstractions train_cfr

test_cards: test_cards.cpp card_utils.h
	$(CXX) $(CXXFLAGS) -o test_cards test_cards.cpp

test_abstractions: test_abstractions.cpp abstractions.h card_utils.h
	$(CXX) $(CXXFLAGS) -o test_abstractions test_abstractions.cpp

train_cfr: main.cpp cfr_engine.h abstractions.h card_utils.h
	$(CXX) $(CXXFLAGS) -O3 -o train_cfr main.cpp

clean:
	rm -f test_cards test_abstractions train_cfr

.PHONY: all clean
