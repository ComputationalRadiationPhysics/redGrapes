/* Copyright 2019-2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <mutex>
#include <functional>
#include <vector>
#include <optional>

#include <boost/context/continuation.hpp>

#include <redGrapes/thread/thread_local.hpp>

namespace redGrapes
{

struct TaskImplBase
{
    virtual ~TaskImplBase() {};
    virtual void run() = 0;

    bool finished;

    TaskImplBase()
        : finished( false )
    {
    }
    
    bool operator() ()
    {
        thread::scope_level = scope_level;

        if( ! resume_cont )
            resume_cont = boost::context::callcc(
                [this]( boost::context::continuation && c )
                {
                    this->yield_cont = std::move( c );
                    this->run();

                    finished = true;
                    
                    return std::move( this->yield_cont );
                }
            );
        else
            resume_cont = resume_cont->resume();

        return finished;
    }

    void yield( )
    {
        yield_cont = yield_cont.resume();
    }

    unsigned int scope_level;

private:
    boost::context::continuation yield_cont;
    std::optional< boost::context::continuation > resume_cont;
};

// TODO: just use std::function
template< typename NullaryCallable >
struct FunctorTask : TaskImplBase
{
    FunctorTask( NullaryCallable && impl )
        : impl( std::move(impl) )
    {}

    ~FunctorTask(){}

    void run()
    {
        this->impl();
    }

private:
    NullaryCallable impl;
};

} // namespace redGrapes
