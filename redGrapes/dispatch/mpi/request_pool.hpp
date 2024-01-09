/* Copyright 2019-2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <redGrapes/scheduler/event.hpp>

#include <mpi.h>

#include <map>
#include <memory>
#include <mutex>

namespace redGrapes
{
    namespace dispatch
    {
        namespace mpi
        {

            struct RequestPool
            {
                std::mutex mutex;

                std::vector<MPI_Request> requests;
                std::vector<scheduler::EventPtr> events;
                std::vector<std::shared_ptr<MPI_Status>> statuses;

                RequestPool()
                {
                }

                /*!
                 * Tests all currently active MPI requests
                 * and notifies the corresponding events if the requests finished
                 */
                void poll()
                {
                    std::lock_guard<std::mutex> lock(mutex);

                    if(!requests.empty())
                    {
                        int outcount;
                        std::vector<int> indices(requests.size());
                        std::vector<MPI_Status> out_statuses(requests.size());

                        MPI_Testsome(requests.size(), requests.data(), &outcount, indices.data(), out_statuses.data());

                        for(int i = 0; i < outcount; ++i)
                        {
                            int idx = indices[i];

                            // write status
                            *(this->statuses[idx]) = out_statuses[i];

                            // finish task waiting for request
                            events[idx].notify();

                            requests.erase(requests.begin() + idx);
                            statuses.erase(statuses.begin() + idx);
                            events.erase(events.begin() + idx);

                            for(int j = i; j < outcount; ++j)
                                if(indices[j] > idx)
                                    indices[j]--;
                        }
                    }
                }

                /*!
                 * Adds a new MPI request to the pool and
                 * yields until the request is done. While waiting
                 * for this request, other tasks will be executed.
                 *
                 * @param request The MPI request to wait for
                 * @return the resulting MPI status of the request
                 */
                MPI_Status get_status(MPI_Request request)
                {
                    auto status = memory::alloc_shared<MPI_Status>();
                    auto event = *create_event();

                    // SPDLOG_TRACE("MPI RequestPool: status event = {}", (void*)event.get());

                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        requests.push_back(request);
                        events.push_back(event);
                        statuses.push_back(status);
                    }

                    yield(event);

                    return *status;
                }
            };

        } // namespace mpi
    } // namespace dispatch
} // namespace redGrapes
