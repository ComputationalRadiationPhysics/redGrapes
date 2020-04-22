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
    virtual void run() = 0;

    void operator() ()
    {
        if( ! resume_cont )
        {
            resume_cont = boost::context::callcc(
                [this]( boost::context::continuation && c )
                {
                    this->yield_cont = std::move(c);

                    for( auto & f : before_hooks )
                        f();

                    this->run();

                    for( auto & f : after_hooks )
                        f();

                    return std::move( this->yield_cont );
                }
            );
        }
        else
        {
            this->resume_cont = this->resume_cont->resume_with([this](auto && c)
                                           {
                                               this->yield_cont = std::move( c );
                                               for( auto & f : resume_hooks )
                                                 f();

                                               return std::move( this->yield_cont );
                                           });
        }
    }

    void yield( unsigned int event_id )
    {
        this->yield_cont = this->yield_cont.resume_with([this, event_id](auto && c){
                                         for( auto & f : pause_hooks )
                                             f( event_id );
                                         return std::move(c);
                                     });
    }

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

    void run()
    {
        this->impl();
    }

private:
    NullaryCallable impl;
};

} // namespace redGrapes
