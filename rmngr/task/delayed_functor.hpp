/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <future>
#include <type_traits>

namespace rmngr
{

template < typename NullaryCallable >
class DelayedFunctor
{
private:
    using Result = typename std::result_of<NullaryCallable()>::type;

public:
    DelayedFunctor(NullaryCallable && impl)
        : impl(std::move(impl)) {}

    void operator() (void)
    {
        set_promise(this->result, this->impl);
    }

    std::future<Result> get_future(void)
    {
        return this->result.get_future();
    }

private:
    NullaryCallable impl;
    std::promise<Result> result;

    template <typename T, typename F>
    static void set_promise (std::promise<T>& p, F& func)
    {
        p.set_value(func());
    }

    template <typename F>
    static void set_promise (std::promise<void>& p, F& func)
    {
        func();
        p.set_value();
    }
}; // class DelayedFunctor

template< typename NullaryCallable >
auto make_delayed_functor( NullaryCallable && f )
{
    return DelayedFunctor< NullaryCallable >(std::move(f));
}

} // namespace rmngr
