
SRCS = main.cpp queue.hpp resource.hpp functor.hpp

test: main
	./$^ > gr

graph.png: test
	dot gr -Tpng -o $@

style: $(SRCS)
	astyle --style=allman --indent-classes --indent-switches $^
