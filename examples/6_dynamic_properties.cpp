
#include <iostream>
#include <thread>
#include <chrono>

#include <rmngr/resource/fieldresource.hpp>
#include <rmngr/scheduler/scheduler.hpp>
#include <rmngr/scheduler/resource.hpp>
#include <rmngr/scheduler/dispatch.hpp>
#include <rmngr/scheduler/fifo.hpp>

using Scheduler = rmngr::Scheduler<
    boost::mpl::vector<
        rmngr::ResourceUserPolicy,
        rmngr::DispatchPolicy< rmngr::FIFO >
    >
>;

Scheduler * scheduler;
rmngr::FieldResource<1> field;

void fun1_impl( int x )
{
    std::cout << "Access " << x << std::endl;
    std::this_thread::sleep_for( std::chrono::seconds(1) );
}

void fun1_prop( Scheduler::Schedulable& s, int x )
{
    s.proto_property< rmngr::ResourceUserPolicy >().access_list =
    {
       field.write({{x,x}}),
    };
}

int main( int, char*[] )
{
    scheduler = new Scheduler(8);
    auto queue = scheduler->get_main_queue();

    auto fun1_proto = scheduler->make_proto( &fun1_impl, &fun1_prop );
    auto fun1 = queue.make_functor( fun1_proto );

    fun1( 1 );
    fun1( 2 );
    fun1( 3 );

    fun1( 1 );
    fun1( 2 );
    fun1( 4 );

    delete scheduler;
    return 0;
}

