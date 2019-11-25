/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <iostream>

#include <redGrapes/manager.hpp>

int fun1_impl (int x)
{
    return x*x;
}

int main()
{
    redGrapes::Manager<> mgr( 1 /* number of threads */ );

    auto fun = mgr.make_functor(&fun1_impl);
    std::cout << "fun(2) = " << fun(2).get() << std::endl;

    return 0;
}

