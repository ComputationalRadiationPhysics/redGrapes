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

namespace redGrapes
{

struct TaskBase
{
    bool finished;
    bool enable_stack_switching;

    virtual ~TaskBase() {}
    TaskBase() : finished(false), enable_stack_switching(false) {}

    virtual void run() = 0;

    std::optional< scheduler::EventPtr > operator() ()
    {
        if( enable_stack_switching )
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
        }
        else
        {
            this->run();
        }

        return event;
    }

    void yield( scheduler::EventPtr event )
    {
        this->event = event;

        if( enable_stack_switching )
        {
            std::optional< boost::context::continuation > old_yield;
            this->yield_cont.swap( old_yield );

            boost::context::continuation new_yield = old_yield->resume();

            std::lock_guard< std::mutex > lock( yield_cont_mutex );
            if( ! yield_cont )
                yield_cont = std::move(new_yield);
            // else: yield_cont already been set by another thread running this task
        }
        else
        {
            spdlog::error("called yield in task without stack switching!");
        }
    }

    std::optional< scheduler::EventPtr > event;

private:
    std::mutex yield_cont_mutex;

    std::optional< boost::context::continuation > yield_cont;
    std::optional< boost::context::continuation > resume_cont;
};

} // namespace redGrapes
