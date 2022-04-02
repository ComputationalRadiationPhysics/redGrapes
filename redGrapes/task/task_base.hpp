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

#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/task/itask.hpp>
#include <redGrapes/dispatch/thread/thread_local.hpp>

namespace redGrapes
{

struct TaskBase : ITask
{
    bool finished;

    virtual ~TaskBase() {}
    TaskBase() : finished(false) {}

    std::optional< std::shared_ptr<scheduler::Event> > operator() ()
    {
        if(!resume_cont)
            resume_cont = boost::context::callcc(
                [this](boost::context::continuation&& c)
                {
                    {
                        std::lock_guard< std::mutex > lock( yield_cont_mutex );
                        this->yield_cont = std::move(c);
                    }

                    this->run();
                    this->event = std::nullopt;

                    std::optional< boost::context::continuation > yield_cont;

                    {
                        std::lock_guard< std::mutex > lock( yield_cont_mutex );
                        this->yield_cont.swap(yield_cont);
                    }

                    return std::move(*yield_cont);
                });
        else
            resume_cont = resume_cont->resume();

        return event;
    }

    void yield( std::shared_ptr<scheduler::Event> event )
    {
        this->event = event;

        std::optional< boost::context::continuation > old_yield;
        this->yield_cont.swap( old_yield );

        boost::context::continuation new_yield = old_yield->resume();

        std::lock_guard< std::mutex > lock( yield_cont_mutex );
        if( ! yield_cont )
            yield_cont = std::move(new_yield);
        // else: yield_cont already been set by another thread running this task
    }

    std::optional< std::shared_ptr<scheduler::Event> > event;

private:
    std::mutex yield_cont_mutex;

    std::optional< boost::context::continuation > yield_cont;
    std::optional< boost::context::continuation > resume_cont;
};

} // namespace redGrapes
