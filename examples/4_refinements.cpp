
#include <thread>
#include <chrono>
#include <iostream>

#include <rmngr/property/resource.hpp>
#include <rmngr/property/inherit.hpp>
#include <rmngr/manager.hpp>

using Properties = rmngr::TaskProperties<
    rmngr::ResourceProperty
>;

int main( int, char*[] )
{
    rmngr::Manager<
        Properties,
        rmngr::ResourceEnqueuePolicy
    > mgr( 4 );

    auto fun1 = mgr.make_functor(
        [&mgr]
        {
            std::cout << "f1 on thread " << rmngr::thread::id << "..." << std::endl;

            mgr.emplace_task(
                []{
                    std::cout << "Refinement 1 on thread " << rmngr::thread::id << std::endl;
                    std::this_thread::sleep_for( std::chrono::seconds(1) );
                });

            mgr.emplace_task(
                []{
                    std::cout << "Refinement 2 on thread " << rmngr::thread::id << std::endl;
                    std::this_thread::sleep_for( std::chrono::seconds(1) );
                });
        }
    );

    fun1();
    fun1();

    return 0;
}

