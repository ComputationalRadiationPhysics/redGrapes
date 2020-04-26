/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <mutex>
#include <functional>
#include <vector>

#include <boost/context/continuation.hpp>
#include <akrzemi/optional.hpp>

namespace redGrapes
{

struct TaskImplBase
{
    virtual ~TaskImplBase() {};
    virtual void run() = 0;

    void operator() ()
    {
        if( ! resume_cont )
        {
            finished = false;
            resume_cont = boost::context::callcc(
                [this]( boost::context::continuation && c )
                {
                    this->yield_cont = std::move(c);

                    for( auto & f : before_hooks )
                        f();

                    this->run();

                    for( auto & f : after_hooks )
                        f();

                    finished = true;

                    return std::move( this->yield_cont );
                }
            );
        }
        else
        {
            for( auto & f : resume_hooks )
                f();

            resume_cont = resume_cont->resume();
        }

        if( ! finished )
        {
            for( auto & f : pause_hooks )
                f( event_id );
        }
    }

    void yield( unsigned int event_id )
    {
        this->event_id = event_id;
        yield_cont = yield_cont.resume();
    }

    bool finished;
    unsigned int event_id;
    
    std::vector<std::function<void()>> before_hooks;
    std::vector<std::function<void()>> after_hooks;

    std::vector<std::function<void(unsigned int)>> pause_hooks;
    std::vector<std::function<void()>> resume_hooks;

private:
    boost::context::continuation yield_cont;
    std::experimental::optional< boost::context::continuation > resume_cont;
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
