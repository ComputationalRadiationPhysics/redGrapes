/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <mutex>
#include <functional>

namespace redGrapes
{

struct TaskImplBase
{
    virtual void run() = 0;

    void operator() ()
    {
        for( auto & f : before_hooks )
            f();
        run();
        for( auto & f : after_hooks )
            f();
    }

    std::vector<std::function<void()>> before_hooks;
    std::vector<std::function<void()>> after_hooks;
};

template< typename NullaryCallable >
struct FunctorTask : TaskImplBase
{
    FunctorTask( NullaryCallable && impl )
        : impl( std::move(impl) )
    {}

    void run()
    {
        this->impl();
    }

private:
    NullaryCallable impl;
};

} // namespace redGrapes
