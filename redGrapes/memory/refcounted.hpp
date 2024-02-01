
#pragma once

#include <atomic>

namespace redGrapes
{

    namespace memory
    {

        template<typename Derived, typename Deleter, typename Refcount = uint64_t>
        struct Refcounted
        {
            std::atomic<Refcount> refcount;

            Refcounted() : refcount(0)
            {
            }

            inline void acquire()
            {
                Refcount old_refcount = refcount.fetch_add(1);
            }

            inline bool release()
            {
                Refcount old_refcount = refcount.fetch_sub(1);
                return old_refcount == 0;
            }

            struct Guard
            {
            private:
                std::atomic<Derived*> ptr;

            public:
                inline Guard() : ptr(nullptr)
                {
                }

                inline Guard(Derived* ptr) : ptr(ptr)
                {
                }

                inline Guard(Guard const& other)
                {
                    acquire(other.ptr.load());
                }

                inline Guard(Guard&& other) : ptr(other.ptr.load())
                {
                    other.ptr = nullptr;
                }

                inline Guard& operator=(Guard const& other)
                {
                    release();
                    acquire(other.ptr.load());
                    return *this;
                }

                inline Guard& operator=(Guard&& other)
                {
                    release();
                    ptr = other.ptr;
                    other.ptr = nullptr;
                    return *this;
                }

                bool compare_exchange_strong(Derived* expected_ptr, Guard new_guard)
                {
                    Derived* desired_ptr = new_guard.ptr.load();
                    new_guard.ptr = expected_ptr;
                    return ptr.compare_exchange_strong(expected_ptr, desired_ptr);
                }

                inline Derived* get() const
                {
                    return ptr.load();
                }

                inline Derived& operator*() const
                {
                    return *ptr.load();
                }

                inline Derived* operator->() const
                {
                    return ptr.load();
                }

                inline bool operator==(Guard const& other) const
                {
                    return ptr == other.ptr;
                }

                inline bool operator!=(Guard const& other) const
                {
                    return ptr != other.ptr;
                }

                inline operator bool() const
                {
                    return ptr.load();
                }

                inline void acquire(Derived* nw_ptr)
                {
                    ptr = nw_ptr;
                    if(nw_ptr)
                        nw_ptr->acquire();
                }

                inline void release()
                {
                    Derived* p = ptr.load();
                    if(p)
                        if(p->release())
                            Deleter{}(p);
                }

                ~Guard()
                {
                    release();
                }
            };
        };

    } // namespace memory

} // namespace redGrapes
