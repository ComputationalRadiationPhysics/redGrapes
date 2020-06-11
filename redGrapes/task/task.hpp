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

#include <boost/context/continuation.hpp>
#include <akrzemi/optional.hpp>

#include <redGrapes/thread/thread_local.hpp>

namespace redGrapes
{

struct TaskImplBase
{
    virtual ~TaskImplBase() {};
    virtual void run() = 0;

    void operator() ()
    {
        thread::scope_level = scope_level;

        if( ! resume_cont )
            resume_cont = boost::context::callcc(
                [this]( boost::context::continuation && c )
                {
                    this->yield_cont = std::move(c);
                    this->run();
                    return std::move( this->yield_cont );
                }
            );
        else
            resume_cont = resume_cont->resume();
    }

    void yield( unsigned int event_id )
    {
        this->event_id = event_id;
        yield_cont = yield_cont.resume();
    }

    unsigned int event_id;

    unsigned int scope_level;

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
