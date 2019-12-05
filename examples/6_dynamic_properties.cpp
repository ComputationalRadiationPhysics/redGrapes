/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <iostream>
#include <thread>
#include <chrono>

#include <redGrapes/resource/fieldresource.hpp>
#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/property/resource.hpp>
#include <redGrapes/property/inherit.hpp>
#include <redGrapes/manager.hpp>

using TaskProperties = redGrapes::TaskProperties<
    redGrapes::ResourceProperty
>;

int main( int, char*[] )
{
    redGrapes::Manager<
        TaskProperties,
        redGrapes::ResourceEnqueuePolicy
    > mgr( 4 );

    redGrapes::FieldResource<1> field;

    auto fun1 = mgr.make_functor(
        []( int x )
        {
            std::cout << "Access " << x << std::endl;
            std::this_thread::sleep_for( std::chrono::seconds(1) );            
        },
        [field]( int x )
        {
            return TaskProperties::Builder().resources({ field.write({{x,x}}) });
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

