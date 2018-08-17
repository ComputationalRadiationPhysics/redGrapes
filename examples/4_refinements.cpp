
#include <thread>
#include <chrono>

#include <iostream>
#include <rmngr/scheduler/scheduler.hpp>
#include <rmngr/scheduler/resource.hpp>
#include <rmngr/scheduler/dispatch.hpp>
#include <rmngr/scheduler/fifo.hpp>
#include <rmngr/scheduler/graphviz.hpp>

using Scheduler =
    rmngr::Scheduler<
        boost::mpl::vector<
            rmngr::ResourceUserPolicy,
          //rmngr::GraphvizWriter< rmngr::DispatchPolicy<rmngr::FIFO>::RuntimeProperty>,
            rmngr::DispatchPolicy< rmngr::FIFO >
        >
    >;

Scheduler * scheduler;

void r1_impl(void)
{
    std::cout << "Refinement 1 on thread " << rmngr::thread::id << std::endl;
    std::this_thread::sleep_for( std::chrono::seconds(1) );
}

void r2_impl(void)
{
    std::cout << "Refinement 2 on thread " << rmngr::thread::id << std::endl;
    std::this_thread::sleep_for( std::chrono::seconds(1) );
}

void f1_impl(void)
{
    std::cout << "f1 on thread " << rmngr::thread::id << "..." << std::endl;

    // get queue for currently running functor
    auto queue = scheduler->get_current_queue();

    // functors as children of current
    auto r1 = queue.make_functor( scheduler->make_proto( &r1_impl ) );
    auto r2 = queue.make_functor( scheduler->make_proto( &r2_impl ) );

    r1();
    r2();
}

int main( int, char*[] )
{
    scheduler = new Scheduler( 16 );

    auto queue = scheduler->get_main_queue();

    auto fun1_proto = scheduler->make_proto( &f1_impl );
    auto fun1 = queue.make_functor(fun1_proto);

    fun1();
    fun1();

    delete scheduler;
    return 0;
}

