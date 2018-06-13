
#include <thread>
#include <chrono>

#include <iostream>
#include <rmngr/resource/ioresource.hpp>
#include <rmngr/scheduler/scheduler.hpp>
#include <rmngr/scheduler/resource.hpp>
#include <rmngr/scheduler/dispatch.hpp>
#include <rmngr/scheduler/fifo.hpp>

using Scheduler =
rmngr::Scheduler<
    boost::mpl::vector<
      rmngr::ResourceUserPolicy,
      rmngr::DispatchPolicy< rmngr::FIFO >
    >
>;

Scheduler* scheduler;
rmngr::IOResource a;

void f1_impl(void)
{
    std::cout << "f1 writes A" << std::endl;
    std::this_thread::sleep_for( std::chrono::seconds(1) );

    std::cout << "f1 now only reads A" << std::endl;
    scheduler->update_property< rmngr::ResourceUserPolicy >(std::vector<rmngr::ResourceAccess>{a.read()});
    std::this_thread::sleep_for( std::chrono::seconds(1) );

    std::cout << "f1 done" << std::endl;
}

void f2_impl(void)
{
    std::cout << "f2 reads A" << std::endl;
    std::cout << "f2 done" << std::endl;
}

int main( int, char*[] )
{
    scheduler = new Scheduler(8);
    auto queue = scheduler->get_main_queue();

    auto fun1_proto = scheduler->make_proto( &f1_impl );
    auto fun2_proto = scheduler->make_proto( &f2_impl );

    scheduler->proto_property<rmngr::ResourceUserPolicy>(fun1_proto).access_list = {a.write()};
    scheduler->proto_property<rmngr::ResourceUserPolicy>(fun2_proto).access_list = {a.read()};

    auto fun1 = queue.make_functor( fun1_proto );
    auto fun2 = queue.make_functor( fun2_proto );

    fun1();
    fun2();

    delete scheduler;
    return 0;
}

