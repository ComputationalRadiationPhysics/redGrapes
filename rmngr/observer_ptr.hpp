
/**
 * @file rmngr/observer_ptr.hpp
 */

#pragma once

#include <memory> // unique_ptr<>

namespace rmngr
{

/**
 * @class observer_ptr
 *
 * Wraps a raw pointer and can be casted to references and pointers.
 * Implements no real functionality, only useful to indicate a borrowed reference
 * instead of ownership.
 */
template <typename T>
class observer_ptr
{
    public:
        observer_ptr()
            : ptr(nullptr) {}

        observer_ptr(T& ref)
            : ptr(&ref) {}

        observer_ptr(T* ptr_)
            : ptr(ptr_) {}

        observer_ptr(std::unique_ptr<T> const& ptr_)
            : ptr(ptr_.get()) {}

        observer_ptr<T>& operator=(observer_ptr<T> const& a)
        {
            this->ptr = a.ptr;
            return *this;
        }

        operator bool () const
        {
            return (this->ptr != nullptr);
        }

        operator T& () const
        {
            return *this->ptr;
        }

        operator T* const () const
        {
            return this->ptr;
        }

        T* operator-> (void) const
        {
            return ptr;
        }

        friend bool operator==(observer_ptr<T> const& a, observer_ptr<T> const& b)
        {
            return a.ptr == b.ptr;
        }

        friend bool operator<(observer_ptr<T> const& a, observer_ptr<T> const& b)
        {
            return a.ptr < b.ptr;
        }

    private:
        T* ptr;
}; // class observer_ptr

}; // namespace rmngr

