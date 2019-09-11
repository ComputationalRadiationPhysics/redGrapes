
#include <iostream>
#include <thread>
#include <chrono>

#include <rmngr/resource/fieldresource.hpp>
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

    rmngr::FieldResource<1> field;

    auto fun1 = mgr.make_functor(
        []( int x )
        {
            std::cout << "Access " << x << std::endl;
            std::this_thread::sleep_for( std::chrono::seconds(1) );            
        },
        [field]( int x )
        {
            return Properties::Builder().resources({ field.write({{x,x}}) });
        }
    );

    fun1( 1 );
    fun1( 2 );
    fun1( 3 );

    fun1( 1 );
    fun1( 2 );
    fun1( 4 );

    return 0;
}

