
#include <iostream>
#include <thread>
#include <chrono>

#include <rmngr/ioresource.hpp>
#include <rmngr/precedence_graph.hpp>
#include <rmngr/scheduling_context.hpp>

int main(void)
{
    rmngr::SchedulingContext context(16);

    auto main_queue = context.get_main_queue();

    rmngr::IOResource a;
    rmngr::IOResource b;

    auto read_a = main_queue.make_functor(
        context.make_proto(
            []()
            {
                std::cout << "Read from A" << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            },
            {a.read()}
      ));

    auto write_a = main_queue.make_functor(
        context.make_proto(
            []()
            {
                std::cout << "Write to A" << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            },
            {a.write()}
    ));

    auto write_b = main_queue.make_functor(
        context.make_proto(
            []()
            {
                std::cout << "Write to B" << std::endl;
            },
            {b.write()}
    ));

    auto read_ab = main_queue.make_functor(
        context.make_proto(
            []()
            {
                std::cout << "Read from A & B" << std::endl;
            },
            {a.read(), b.read()}
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

    return 0;
}

