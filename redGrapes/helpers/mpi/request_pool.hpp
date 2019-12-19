/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <mpi.h>
#include <mutex>
#include <map>
#include <redGrapes/resource/ioresource.hpp>

namespace redGrapes
{
namespace helpers
{
namespace mpi
{

template <typename Manager>
struct RequestPool
{
    using EventID = typename Manager::EventID;

    Manager & mgr;
    std::mutex mutex;
    std::map<MPI_Request*, std::pair<EventID, ioresource::WriteGuard<MPI_Status>>> requests;

    RequestPool( Manager & mgr )
        : mgr(mgr)
    {}

    /**
     * Tests all currently active MPI requests
     * and notifies the corresponding events if the requests finished
     */
    void poll()
    {
        std::vector<std::tuple<MPI_Request*, EventID, ioresource::WriteGuard<MPI_Status>>> r;
        {
            std::lock_guard<std::mutex> lock(mutex);
            r.reserve(requests.size());
            for( auto it = requests.begin(); it != requests.end(); ++it )
                r.emplace_back(it->first, it->second.first, it->second.second);
        }

        for( auto request : r )
        {
            int flag = 0;
            MPI_Test(std::get<0>(request), &flag, &(*std::get<2>(request)));
            if( flag )
            {
                std::lock_guard<std::mutex> lock(mutex);
                requests.erase( std::get<0>(request) );
                mgr.reach_event( std::get<1>(request) );
            }
        }
    }

    /**
     * Adds a new MPI request to the pool and creates a child task
     * that finishes after the request is done
     *
     * @param request new MPI request
     * @return IOResource of the MPI status
     */
    auto wait( MPI_Request * request )
    {
        IOResource< MPI_Status > status;
        mgr.emplace_task(
            [this, request]( auto status )
            {
                std::lock_guard<std::mutex> lock(mutex);
                requests.emplace(request, std::make_pair(*mgr.create_event(), status));
            },
            status.write()
        );

        return status;
    }
};

} // namespace mpi

} // namespace helpers

} // namespace redGrapes

