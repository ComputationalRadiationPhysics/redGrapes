
#include <iostream>
#include <thread>
#include <chrono>

#include <rmngr/resource/ioresource.hpp>
#include <rmngr/scheduler/scheduler.hpp>
#include <rmngr/scheduler/resource.hpp>
#include <rmngr/scheduler/dispatch.hpp>
#include <rmngr/scheduler/fifo.hpp>

int main(void)
{
    rmngr::Scheduler<
        boost::mpl::vector<
            rmngr::ResourceUserPolicy,
            rmngr::DispatchPolicy< rmngr::FIFO >
        >
    > scheduler;

    auto main_queue = scheduler.get_main_queue();

    rmngr::IOResource a;
    rmngr::IOResource b;

    auto read_a_proto = scheduler.make_proto(
        []()
        {
            std::cout << "Read from A" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        });
    scheduler.proto_property<rmngr::ResourceUserPolicy>(read_a_proto).access_list = {a.read()};
    auto read_a = main_queue.make_functor( read_a_proto );

    auto write_a_proto = scheduler.make_proto(
        []()
        {
            std::cout << "Write to A" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        });
    scheduler.proto_property<rmngr::ResourceUserPolicy>(write_a_proto).access_list = {a.write()};
    auto write_a = main_queue.make_functor( write_a_proto );

    auto write_b_proto = scheduler.make_proto(
        []()
        {
            std::cout << "Write to B" << std::endl;
        });
    scheduler.proto_property<rmngr::ResourceUserPolicy>(write_b_proto).access_list = {b.write()};
    auto write_b = main_queue.make_functor( write_b_proto );

    auto read_ab_proto = scheduler.make_proto(
        []()
        {
            std::cout << "Read from A & B" << std::endl;
        });
    scheduler.proto_property<rmngr::ResourceUserPolicy>(read_ab_proto).access_list = {a.read(), b.read()};
    auto read_ab = main_queue.make_functor( read_ab_proto );

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

