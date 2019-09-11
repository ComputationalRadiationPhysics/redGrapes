
#include <thread>
#include <chrono>

#include <iostream>
#include <rmngr/resource/ioresource.hpp>
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

    rmngr::IOResource a;

    mgr.emplace_task(
        [&mgr, a]
        {
            std::cout << "f1 writes A" << std::endl;
            std::this_thread::sleep_for( std::chrono::seconds(1) );

            std::cout << "f1 now only reads A" << std::endl;
            mgr.update_properties(
                Properties::Patch::Builder()
                    .remove_resources({ a.write() })
                    .add_resources({ a.read() })
            );
            std::this_thread::sleep_for( std::chrono::seconds(1) );

            std::cout << "f1 done" << std::endl; 
        },
        Properties::Builder().resources({ a.write() })
    );

    mgr.emplace_task(
        []
        {
            std::cout << "f2 reads A" << std::endl;
            std::cout << "f2 done" << std::endl;
        },
        Properties::Builder().resources({ a.read() })
    );
    
    return 0;
}

