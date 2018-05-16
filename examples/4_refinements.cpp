
#include <thread>
#include <chrono>

#include <iostream>
#include <rmngr/ioresource.hpp>
#include <rmngr/scheduling_context.hpp>

rmngr::SchedulingContext* context;

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
    auto queue = context->get_current_queue();

    // functors as children of current
    auto r1 = queue.make_functor( context->make_proto( &r1_impl ) );
    auto r2 = queue.make_functor( context->make_proto( &r2_impl ) );

    r1();
    r2();
}

int main( int, char*[] )
{
    context = new rmngr::SchedulingContext(8);
    auto queue = context->get_main_queue();

    rmngr::IOResource a, b;

    auto fun1_proto = context->make_proto( &f1_impl, {a.read()} );
    auto fun1 = queue.make_functor(fun1_proto);

    fun1();
    fun1();

    delete context;
    return 0;
}

