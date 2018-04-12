
#include <iostream>
#include <thread>
#include <chrono>

#include <rmngr/ioresource.hpp>
#include <rmngr/precedence_graph.hpp>
#include <rmngr/scheduling_context.hpp>

int main(void)
{
    using Context = rmngr::SchedulingContext<1>;
    using Graph = typename Context::Graph;
    rmngr::QueuedPrecedenceGraph<Graph, rmngr::ResourceUser> main_refinement;
    Context context(main_refinement);
    rmngr::IOResource a;
    rmngr::IOResource b;

    auto main_queue = rmngr::make_functor_queue(main_refinement);

    auto read_a = main_queue.make_functor(
        context.make_proto(
            []()
            {
                std::cout << "Read from A" << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            },
            {a.read()},
            "Ar"
      ));

    auto write_a = main_queue.make_functor(
        context.make_proto(
            []()
            {
                std::cout << "Write to A" << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            },
            {a.write()},
            "Aw"
    ));

    auto write_b = main_queue.make_functor(
        context.make_proto(
            []()
            {
                std::cout << "Write to B" << std::endl;
            },
            {b.write()},
            "Bw"
    ));

    auto read_ab = main_queue.make_functor(
        context.make_proto(
            []()
            {
                std::cout << "Read from A & B" << std::endl;
            },
            {a.read(), b.read()},
            "Ar, Br"
    ));

    for(int i = 0; i < 1; ++i)
    {
        write_a();
        write_a();
        read_a();
        read_a();
        write_b();
        read_ab();
    }

    context.scheduler.update_schedule();

    return 0;
}

