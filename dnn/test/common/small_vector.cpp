/**
 * \file dnn/test/common/small_vector.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
//===- llvm/unittest/ADT/SmallVectorTest.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// SmallVector unit tests.
//
//===----------------------------------------------------------------------===//
/**
 * \file common/small_vector.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 */

#include "megdnn/thin/small_vector.h"

#include <gtest/gtest.h>
#include <cstdarg>
#include <list>
#include <unordered_set>

using namespace megdnn;

namespace {

/// A helper class that counts the total number of constructor and
/// destructor calls.
class Constructable {
private:
    static int num_constructor_calls;
    static int num_move_constructor_calls;
    static int num_copy_constructor_calls;
    static int num_deconstructor_calls;
    static int num_assignment_calls;
    static int num_move_assignment_calls;
    static int num_copy_assignment_calls;

    static std::unordered_set<const Constructable*> destroyed_mem;

    bool m_constructed;
    int m_value;

    int m_id;

public:
    Constructable() : m_constructed(true), m_value(0) {
        ++num_constructor_calls;
        m_id = num_constructor_calls;
        destroyed_mem.erase(this);
    }

    Constructable(int val) : m_constructed(true), m_value(val) {
        ++num_constructor_calls;
        m_id = num_constructor_calls;
        destroyed_mem.erase(this);
    }

    Constructable(const Constructable& src) : m_constructed(true) {
        m_value = src.m_value;
        ++num_constructor_calls;
        m_id = num_constructor_calls;
        EXPECT_TRUE(destroyed_mem.find(&src) == destroyed_mem.end());
        destroyed_mem.erase(this);
    }

    Constructable(Constructable&& src) : m_constructed(true) {
        m_value = src.m_value;
        ++num_constructor_calls;
        ++num_move_constructor_calls;
        m_id = num_constructor_calls;
        EXPECT_TRUE(destroyed_mem.find(&src) == destroyed_mem.end());
        destroyed_mem.erase(this);
    }

    ~Constructable() {
        EXPECT_TRUE(m_constructed);
        ++num_deconstructor_calls;
        m_constructed = false;
        destroyed_mem.insert(this);
    }

    Constructable& operator=(const Constructable& src) {
        EXPECT_TRUE(m_constructed);
        m_value = src.m_value;
        ++num_assignment_calls;
        ++num_copy_assignment_calls;
        m_id = src.m_id;
        EXPECT_TRUE(destroyed_mem.find(&src) == destroyed_mem.end());
        return *this;
    }

    Constructable& operator=(Constructable&& src) {
        EXPECT_TRUE(m_constructed);
        m_value = src.m_value;
        ++num_assignment_calls;
        ++num_move_assignment_calls;
        m_id = src.m_id;
        return *this;
    }

    int get_value() const { return abs(m_value); }

    static void reset() {
        num_constructor_calls = 0;
        num_move_constructor_calls = 0;
        num_copy_constructor_calls = 0;
        num_deconstructor_calls = 0;
        num_assignment_calls = 0;
        num_move_assignment_calls = 0;
        num_copy_assignment_calls = 0;
        destroyed_mem.clear();
    }

    static int get_num_constructor_calls() { return num_constructor_calls; }

    static int get_num_move_constructor_calls() {
        return num_move_constructor_calls;
    }

    static int get_num_copy_constructor_calls() {
        return num_copy_constructor_calls;
    }

    static int get_num_destructor_calls() { return num_deconstructor_calls; }

    static int get_num_assignment_calls() { return num_assignment_calls; }

    static int get_num_move_assignment_calls() {
        return num_move_assignment_calls;
    }

    static int get_num_copy_assignment_calls() {
        return num_copy_assignment_calls;
    }

    bool operator==(const Constructable& rhs) const {
        return this->get_value() == rhs.get_value();
    }

    bool operator!=(const Constructable& rhs) const {
        return this->get_value() != rhs.get_value();
    }
};

int Constructable::num_constructor_calls;
int Constructable::num_copy_constructor_calls;
int Constructable::num_move_constructor_calls;
int Constructable::num_deconstructor_calls;
int Constructable::num_assignment_calls;
int Constructable::num_copy_assignment_calls;
int Constructable::num_move_assignment_calls;
std::unordered_set<const Constructable*> Constructable::destroyed_mem;

struct NonCopyable {
    NonCopyable() {}
    NonCopyable(NonCopyable&&) {}
    NonCopyable& operator=(NonCopyable&&) { return *this; }

private:
    NonCopyable(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;
};

struct MoveOnly {
    int val;

    MoveOnly(int v) : val{v} {}
    MoveOnly(MoveOnly&& rhs) : val{rhs.val} { rhs.val = 0; }
    MoveOnly& operator=(MoveOnly&& rhs) {
        val = rhs.val;
        rhs.val = 0;
        return *this;
    }

private:
    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;
};

__attribute__((__unused__)) void compile_test() {
    SmallVector<NonCopyable, 0> v;
    v.resize(42);
}

class SmallVectorTestBase : public testing::Test {
protected:
    void SetUp() override { Constructable::reset(); }

    template <typename VectorT>
    void assert_empty(VectorT& v) {
        // Size tests
        EXPECT_EQ(0u, v.size());
        EXPECT_TRUE(v.empty());

        // Iterator tests
        EXPECT_TRUE(v.begin() == v.end());
    }

    // Assert that v contains the specified values, in order.
    template <typename VectorT>
    void assert_values_in_order(VectorT& v, size_t size, ...) {
        EXPECT_EQ(size, v.size());

        va_list ap;
        va_start(ap, size);
        for (size_t i = 0; i < size; ++i) {
            int m_value = va_arg(ap, int);
            EXPECT_EQ(m_value, v[i].get_value());
        }

        va_end(ap);
    }

    // Generate a sequence of values to initialize the vector.
    template <typename VectorT>
    void make_sequence(VectorT& v, int start, int end) {
        for (int i = start; i <= end; ++i) {
            v.push_back(Constructable(i));
        }
    }
};

// Test fixture class
template <typename VectorT>
class SmallVectorTest : public SmallVectorTestBase {
protected:
    VectorT the_vector;
    VectorT other_vector;
};

typedef ::testing::Types<
        SmallVector<Constructable, 0>, SmallVector<Constructable, 1>,
        SmallVector<Constructable, 2>, SmallVector<Constructable, 4>,
        SmallVector<Constructable, 5>>
        SmallVectorTestTypes;
TYPED_TEST_CASE(SmallVectorTest, SmallVectorTestTypes);
// Constructor test.
TYPED_TEST(SmallVectorTest, ConstructorNonIterTest) {
    SCOPED_TRACE("ConstructorTest");
    this->the_vector = SmallVector<Constructable, 2>(2, 2);
    this->assert_values_in_order(this->the_vector, 2u, 2, 2);
}

// Constructor test.
TYPED_TEST(SmallVectorTest, ConstructorIterTest) {
    SCOPED_TRACE("ConstructorTest");
    int arr[] = {1, 2, 3};
    this->the_vector =
            SmallVector<Constructable, 4>(std::begin(arr), std::end(arr));
    this->assert_values_in_order(this->the_vector, 3u, 1, 2, 3);
}

// New vector test.
TYPED_TEST(SmallVectorTest, EmptyVectorTest) {
    SCOPED_TRACE("EmptyVectorTest");
    this->assert_empty(this->the_vector);
    EXPECT_TRUE(this->the_vector.rbegin() == this->the_vector.rend());
    EXPECT_EQ(0, Constructable::get_num_constructor_calls());
    EXPECT_EQ(0, Constructable::get_num_destructor_calls());
}

// Simple insertions and deletions.
TYPED_TEST(SmallVectorTest, PushPopTest) {
    SCOPED_TRACE("PushPopTest");

    // Track whether the vector will potentially have to grow.
    bool require_growth = this->the_vector.capacity() < 3;

    // Push an element
    this->the_vector.push_back(Constructable(1));

    // Size tests
    this->assert_values_in_order(this->the_vector, 1u, 1);
    EXPECT_FALSE(this->the_vector.begin() == this->the_vector.end());
    EXPECT_FALSE(this->the_vector.empty());

    // Push another element
    this->the_vector.push_back(Constructable(2));
    this->assert_values_in_order(this->the_vector, 2u, 1, 2);

    // Insert at beginning
    this->the_vector.insert(this->the_vector.begin(), this->the_vector[1]);
    this->assert_values_in_order(this->the_vector, 3u, 2, 1, 2);

    // Pop one element
    this->the_vector.pop_back();
    this->assert_values_in_order(this->the_vector, 2u, 2, 1);

    // Pop remaining elements
    this->the_vector.pop_back();
    this->the_vector.pop_back();
    this->assert_empty(this->the_vector);

    // Check number of constructor calls. Should be 2 for each list element,
    // one for the argument to push_back, one for the argument to insert,
    // and one for the list element itself.
    if (!require_growth) {
        // Original test expected number is 5, however, after fixing the bug of
        // out of range while inserting element within vector, the
        // CopyConstructor would be called 1 more times.
        EXPECT_EQ(6, Constructable::get_num_constructor_calls());
        EXPECT_EQ(6, Constructable::get_num_destructor_calls());
    } else {
        // If we had to grow the vector, these only have a lower bound, but
        // should
        // always be equal.
        EXPECT_LE(5, Constructable::get_num_constructor_calls());
        EXPECT_EQ(Constructable::get_num_constructor_calls(),
                  Constructable::get_num_destructor_calls());
    }
}

// Clear test.
TYPED_TEST(SmallVectorTest, ClearTest) {
    SCOPED_TRACE("ClearTest");

    this->the_vector.reserve(2);
    this->make_sequence(this->the_vector, 1, 2);
    this->the_vector.clear();

    this->assert_empty(this->the_vector);
    EXPECT_EQ(4, Constructable::get_num_constructor_calls());
    EXPECT_EQ(4, Constructable::get_num_destructor_calls());
}

// Resize smaller test.
TYPED_TEST(SmallVectorTest, ResizeShrinkTest) {
    SCOPED_TRACE("ResizeShrinkTest");

    this->the_vector.reserve(3);
    this->make_sequence(this->the_vector, 1, 3);
    this->the_vector.resize(1);

    this->assert_values_in_order(this->the_vector, 1u, 1);
    EXPECT_EQ(6, Constructable::get_num_constructor_calls());
    EXPECT_EQ(5, Constructable::get_num_destructor_calls());
}

// Resize bigger test.
TYPED_TEST(SmallVectorTest, ResizeGrowTest) {
    SCOPED_TRACE("ResizeGrowTest");

    this->the_vector.resize(2);

    EXPECT_EQ(2, Constructable::get_num_constructor_calls());
    EXPECT_EQ(0, Constructable::get_num_destructor_calls());
    EXPECT_EQ(2u, this->the_vector.size());
}

TYPED_TEST(SmallVectorTest, ResizeWithElementsTest) {
    this->the_vector.resize(2);

    Constructable::reset();

    this->the_vector.resize(4);

    size_t ctors = Constructable::get_num_constructor_calls();
    EXPECT_TRUE(ctors == 2 || ctors == 4);
    size_t movectors = Constructable::get_num_move_constructor_calls();
    EXPECT_TRUE(movectors == 0 || movectors == 2);
    size_t dtors = Constructable::get_num_destructor_calls();
    EXPECT_TRUE(dtors == 0 || dtors == 2);
}

// Resize with fill m_value.
TYPED_TEST(SmallVectorTest, ResizeFillTest) {
    SCOPED_TRACE("ResizeFillTest");

    this->the_vector.resize(3, Constructable(77));
    this->assert_values_in_order(this->the_vector, 3u, 77, 77, 77);
}

// Overflow past fixed size.
TYPED_TEST(SmallVectorTest, OverflowTest) {
    SCOPED_TRACE("OverflowTest");

    // Push more elements than the fixed size.
    this->make_sequence(this->the_vector, 1, 10);

    // Test size and values.
    EXPECT_EQ(10u, this->the_vector.size());
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(i + 1, this->the_vector[i].get_value());
    }

    // Now resize back to fixed size.
    this->the_vector.resize(1);

    this->assert_values_in_order(this->the_vector, 1u, 1);
}

// Iteration tests.
TYPED_TEST(SmallVectorTest, IterationTest) {
    this->make_sequence(this->the_vector, 1, 2);

    // Forward Iteration
    typename TypeParam::iterator it = this->the_vector.begin();
    EXPECT_TRUE(*it == this->the_vector.front());
    EXPECT_TRUE(*it == this->the_vector[0]);
    EXPECT_EQ(1, it->get_value());
    ++it;
    EXPECT_TRUE(*it == this->the_vector[1]);
    EXPECT_TRUE(*it == this->the_vector.back());
    EXPECT_EQ(2, it->get_value());
    ++it;
    EXPECT_TRUE(it == this->the_vector.end());
    --it;
    EXPECT_TRUE(*it == this->the_vector[1]);
    EXPECT_EQ(2, it->get_value());
    --it;
    EXPECT_TRUE(*it == this->the_vector[0]);
    EXPECT_EQ(1, it->get_value());

    // Reverse Iteration
    typename TypeParam::reverse_iterator rit = this->the_vector.rbegin();
    EXPECT_TRUE(*rit == this->the_vector[1]);
    EXPECT_EQ(2, rit->get_value());
    ++rit;
    EXPECT_TRUE(*rit == this->the_vector[0]);
    EXPECT_EQ(1, rit->get_value());
    ++rit;
    EXPECT_TRUE(rit == this->the_vector.rend());
    --rit;
    EXPECT_TRUE(*rit == this->the_vector[0]);
    EXPECT_EQ(1, rit->get_value());
    --rit;
    EXPECT_TRUE(*rit == this->the_vector[1]);
    EXPECT_EQ(2, rit->get_value());
}

// Swap test.
TYPED_TEST(SmallVectorTest, SwapTest) {
    SCOPED_TRACE("SwapTest");

    this->make_sequence(this->the_vector, 1, 2);
    this->make_sequence(this->other_vector, 1, 4);
    std::swap(this->the_vector, this->other_vector);

    this->assert_values_in_order(this->the_vector, 4u, 1, 2, 3, 4);
    this->assert_values_in_order(this->other_vector, 2u, 1, 2);
}

// Symmetric to previoud Swap Test.
TYPED_TEST(SmallVectorTest, SwapReverseTest) {
    SCOPED_TRACE("SwapReverseTest");
    this->make_sequence(this->the_vector, 1, 2);
    this->make_sequence(this->other_vector, 1, 4);
    std::swap(this->other_vector, this->the_vector);
    this->assert_values_in_order(this->the_vector, 4u, 1, 2, 3, 4);
    this->assert_values_in_order(this->other_vector, 2u, 1, 2);
}

// Swap two vectors with different default size N.
TYPED_TEST(SmallVectorTest, SwapSpecificSizeWithoutGrowingTest) {
    SCOPED_TRACE("SwapSpecificSizeWithoutGrowingTest");
    SmallVector<Constructable, 3> other_vector;
    // not grow.
    this->make_sequence(other_vector, 1, 2);
    this->make_sequence(this->the_vector, 1, 3);
    std::swap(other_vector, this->the_vector);
    this->assert_values_in_order(other_vector, 3u, 1, 2, 3);
    this->assert_values_in_order(this->the_vector, 2u, 1, 2);
}

// Swap two vectors with different default size N.
TYPED_TEST(SmallVectorTest, SwapSpecificSizeWithGrowingTest) {
    SCOPED_TRACE("SwapSpecificSizeWithGrowingTest");
    SmallVector<Constructable, 3> other_vector;
    // grow
    this->make_sequence(other_vector, 1, 4);
    this->make_sequence(this->the_vector, 1, 3);
    std::swap(other_vector, this->the_vector);
    this->assert_values_in_order(other_vector, 3u, 1, 2, 3);
    this->assert_values_in_order(this->the_vector, 4u, 1, 2, 3, 4);
}

// Append test
TYPED_TEST(SmallVectorTest, AppendTest) {
    SCOPED_TRACE("AppendTest");

    this->make_sequence(this->other_vector, 2, 3);

    this->the_vector.push_back(Constructable(1));
    this->the_vector.append(this->other_vector.begin(),
                            this->other_vector.end());

    this->assert_values_in_order(this->the_vector, 3u, 1, 2, 3);
}

// Append repeated test
TYPED_TEST(SmallVectorTest, AppendRepeatedTest) {
    SCOPED_TRACE("AppendRepeatedTest");

    this->the_vector.push_back(Constructable(1));
    this->the_vector.append(2, Constructable(77));
    this->assert_values_in_order(this->the_vector, 3u, 1, 77, 77);
}

// Append test
TYPED_TEST(SmallVectorTest, AppendNonIterTest) {
    SCOPED_TRACE("AppendRepeatedTest");

    this->the_vector.push_back(Constructable(1));
    this->the_vector.append(2, 7);
    this->assert_values_in_order(this->the_vector, 3u, 1, 7, 7);
}

struct OutputIterator {
    typedef std::output_iterator_tag iterator_category;
    typedef int value_type;
    typedef int difference_type;
    typedef value_type* pointer;
    typedef value_type& reference;
    operator int() { return 2; }
    operator Constructable() { return 7; }
};

TYPED_TEST(SmallVectorTest, AppendRepeatedNonForwardIterator) {
    SCOPED_TRACE("AppendRepeatedTest");

    this->the_vector.push_back(Constructable(1));
    this->the_vector.append(OutputIterator(), OutputIterator());
    this->assert_values_in_order(this->the_vector, 3u, 1, 7, 7);
}

// Assign test
TYPED_TEST(SmallVectorTest, AssignTest) {
    SCOPED_TRACE("AssignTest");

    this->the_vector.push_back(Constructable(1));
    this->the_vector.assign(2, Constructable(77));
    this->assert_values_in_order(this->the_vector, 2u, 77, 77);
}

// Assign test
TYPED_TEST(SmallVectorTest, AssignRangeTest) {
    SCOPED_TRACE("AssignTest");

    this->the_vector.push_back(Constructable(1));
    int arr[] = {1, 2, 3};
    this->the_vector.assign(std::begin(arr), std::end(arr));
    this->assert_values_in_order(this->the_vector, 3u, 1, 2, 3);
}

// Assign test
TYPED_TEST(SmallVectorTest, AssignNonIterTest) {
    SCOPED_TRACE("AssignTest");

    this->the_vector.push_back(Constructable(1));
    this->the_vector.assign(2, 7);
    this->assert_values_in_order(this->the_vector, 2u, 7, 7);
}

// Move-assign test
TYPED_TEST(SmallVectorTest, MoveAssignTest) {
    SCOPED_TRACE("MoveAssignTest");

    // Set up our vector with a single element, but enough capacity for 4.
    this->the_vector.reserve(4);
    this->the_vector.push_back(Constructable(1));

    // Set up the other vector with 2 elements.
    this->other_vector.push_back(Constructable(2));
    this->other_vector.push_back(Constructable(3));

    // Move-assign from the other vector.
    this->the_vector = std::move(this->other_vector);

    // Make sure we have the right result.
    this->assert_values_in_order(this->the_vector, 2u, 2, 3);

    // Make sure the # of constructor/destructor calls line up. There
    // are two live objects after clearing the other vector.
    this->other_vector.clear();
    EXPECT_EQ(Constructable::get_num_constructor_calls() - 2,
              Constructable::get_num_destructor_calls());

    // There shouldn't be any live objects any more.
    this->the_vector.clear();
    EXPECT_EQ(Constructable::get_num_constructor_calls(),
              Constructable::get_num_destructor_calls());
}

// Erase a single element
TYPED_TEST(SmallVectorTest, EraseTest) {
    SCOPED_TRACE("EraseTest");

    this->make_sequence(this->the_vector, 1, 3);
    const auto& the_const_vector = this->the_vector;
    this->the_vector.erase(the_const_vector.begin());
    this->assert_values_in_order(this->the_vector, 2u, 2, 3);
}

// Erase a range of elements
TYPED_TEST(SmallVectorTest, EraseRangeTest) {
    SCOPED_TRACE("EraseRangeTest");

    this->make_sequence(this->the_vector, 1, 3);
    const auto& the_const_vector = this->the_vector;
    this->the_vector.erase(the_const_vector.begin(),
                           the_const_vector.begin() + 2);
    this->assert_values_in_order(this->the_vector, 1u, 3);
}

// Insert a single element.
TYPED_TEST(SmallVectorTest, InsertTest) {
    SCOPED_TRACE("InsertTest");

    this->make_sequence(this->the_vector, 1, 3);
    typename TypeParam::iterator it = this->the_vector.insert(
            this->the_vector.begin() + 1, Constructable(77));
    EXPECT_EQ(this->the_vector.begin() + 1, it);
    this->assert_values_in_order(this->the_vector, 4u, 1, 77, 2, 3);
}

// Insert a copy of a single element.
TYPED_TEST(SmallVectorTest, InsertCopy) {
    SCOPED_TRACE("InsertTest");

    this->make_sequence(this->the_vector, 1, 3);
    Constructable c(77);
    typename TypeParam::iterator it =
            this->the_vector.insert(this->the_vector.begin() + 1, c);
    EXPECT_EQ(this->the_vector.begin() + 1, it);
    this->assert_values_in_order(this->the_vector, 4u, 1, 77, 2, 3);
}

// Insert repeated elements.
TYPED_TEST(SmallVectorTest, InsertRepeatedTest) {
    SCOPED_TRACE("InsertRepeatedTest");

    this->make_sequence(this->the_vector, 1, 4);
    Constructable::reset();
    auto it = this->the_vector.insert(this->the_vector.begin() + 1, 2,
                                      Constructable(16));
    // Move construct the top element into newly allocated space, and optionally
    // reallocate the whole buffer, move constructing into it.
    // FIXME: This is inefficient, we shouldn't move things into newly allocated
    // space, then move them up/around, there should only be 2 or 4 move
    // constructions here.
    EXPECT_TRUE(Constructable::get_num_move_constructor_calls() == 2 ||
                Constructable::get_num_move_constructor_calls() == 6);
    // Move assign the next two to shift them up and make a gap.
    EXPECT_EQ(1, Constructable::get_num_move_assignment_calls());
    // Copy construct the two new elements from the parameter.
    EXPECT_EQ(2, Constructable::get_num_copy_assignment_calls());
    // All without any copy construction.
    // EXPECT_EQ(0, Constructable::get_num_copy_constructor_calls());
    EXPECT_EQ(this->the_vector.begin() + 1, it);
    this->assert_values_in_order(this->the_vector, 6u, 1, 16, 16, 2, 3, 4);
}

TYPED_TEST(SmallVectorTest, InsertRepeatedNonIterTest) {
    SCOPED_TRACE("InsertRepeatedTest");

    this->make_sequence(this->the_vector, 1, 4);
    Constructable::reset();
    auto it = this->the_vector.insert(this->the_vector.begin() + 1, 2, 7);
    EXPECT_EQ(this->the_vector.begin() + 1, it);
    this->assert_values_in_order(this->the_vector, 6u, 1, 7, 7, 2, 3, 4);
}

TYPED_TEST(SmallVectorTest, InsertRepeatedAtEndTest) {
    SCOPED_TRACE("InsertRepeatedTest");

    this->make_sequence(this->the_vector, 1, 4);
    Constructable::reset();
    auto it = this->the_vector.insert(this->the_vector.end(), 2,
                                      Constructable(16));
    // Just copy construct them into newly allocated space
    // EXPECT_EQ(2, Constructable::get_num_copy_constructor_calls());
    // Move everything across if reallocation is needed.
    EXPECT_TRUE(Constructable::get_num_move_constructor_calls() == 0 ||
                Constructable::get_num_move_constructor_calls() == 4);
    // Without ever moving or copying anything else.
    EXPECT_EQ(0, Constructable::get_num_copy_assignment_calls());
    EXPECT_EQ(0, Constructable::get_num_move_assignment_calls());

    EXPECT_EQ(this->the_vector.begin() + 4, it);
    this->assert_values_in_order(this->the_vector, 6u, 1, 2, 3, 4, 16, 16);
}

TYPED_TEST(SmallVectorTest, InsertRepeatedEmptyTest) {
    SCOPED_TRACE("InsertRepeatedTest");

    this->make_sequence(this->the_vector, 10, 15);

    // Empty insert.
    EXPECT_EQ(this->the_vector.end(),
              this->the_vector.insert(this->the_vector.end(), 0,
                                      Constructable(42)));
    EXPECT_EQ(this->the_vector.begin() + 1,
              this->the_vector.insert(this->the_vector.begin() + 1, 0,
                                      Constructable(42)));
}

// Insert range.
TYPED_TEST(SmallVectorTest, InsertRangeTest) {
    SCOPED_TRACE("InsertRangeTest");

    Constructable arr[3] = {Constructable(77), Constructable(77),
                            Constructable(77)};

    this->make_sequence(this->the_vector, 1, 3);
    Constructable::reset();
    auto it =
            this->the_vector.insert(this->the_vector.begin() + 1, arr, arr + 3);
    // Move construct the top 3 elements into newly allocated space.
    // Possibly move the whole sequence into new space first.
    // FIXME: This is inefficient, we shouldn't move things into newly allocated
    // space, then move them up/around, there should only be 2 or 3 move
    // constructions here.
    EXPECT_TRUE(Constructable::get_num_move_constructor_calls() == 2 ||
                Constructable::get_num_move_constructor_calls() == 5);
    // Copy assign the lower 2 new elements into existing space.
    EXPECT_EQ(2, Constructable::get_num_copy_assignment_calls());
    // Copy construct the third element into newly allocated space.
    // EXPECT_EQ(1, Constructable::get_num_copy_constructor_calls());
    EXPECT_EQ(this->the_vector.begin() + 1, it);
    this->assert_values_in_order(this->the_vector, 6u, 1, 77, 77, 77, 2, 3);
}

TYPED_TEST(SmallVectorTest, InsertRangeAtEndTest) {
    SCOPED_TRACE("InsertRangeTest");

    Constructable arr[3] = {Constructable(77), Constructable(77),
                            Constructable(77)};

    this->make_sequence(this->the_vector, 1, 3);

    // Insert at end.
    Constructable::reset();
    auto it = this->the_vector.insert(this->the_vector.end(), arr, arr + 3);
    // Copy construct the 3 elements into new space at the top.
    // EXPECT_EQ(3, Constructable::get_num_copy_constructor_calls());
    // Don't copy/move anything else.
    EXPECT_EQ(0, Constructable::get_num_copy_assignment_calls());
    // Reallocation might occur, causing all elements to be moved into the new
    // buffer.
    EXPECT_TRUE(Constructable::get_num_move_constructor_calls() == 0 ||
                Constructable::get_num_move_constructor_calls() == 3);
    EXPECT_EQ(0, Constructable::get_num_move_assignment_calls());
    EXPECT_EQ(this->the_vector.begin() + 3, it);
    this->assert_values_in_order(this->the_vector, 6u, 1, 2, 3, 77, 77, 77);
}

TYPED_TEST(SmallVectorTest, InsertEmptyRangeTest) {
    SCOPED_TRACE("InsertRangeTest");

    this->make_sequence(this->the_vector, 1, 3);

    // Empty insert.
    EXPECT_EQ(this->the_vector.end(),
              this->the_vector.insert(this->the_vector.end(),
                                      this->the_vector.begin(),
                                      this->the_vector.begin()));
    EXPECT_EQ(this->the_vector.begin() + 1,
              this->the_vector.insert(this->the_vector.begin() + 1,
                                      this->the_vector.begin(),
                                      this->the_vector.begin()));
}

// Comparison tests.
TYPED_TEST(SmallVectorTest, ComparisonTest) {
    SCOPED_TRACE("ComparisonTest");

    this->make_sequence(this->the_vector, 1, 3);
    this->make_sequence(this->other_vector, 1, 3);

    EXPECT_TRUE(this->the_vector == this->other_vector);
    EXPECT_FALSE(this->the_vector != this->other_vector);

    this->other_vector.clear();
    this->make_sequence(this->other_vector, 2, 4);

    EXPECT_FALSE(this->the_vector == this->other_vector);
    EXPECT_TRUE(this->the_vector != this->other_vector);
}

// Constant vector tests.
TYPED_TEST(SmallVectorTest, ConstVectorTest) {
    const TypeParam const_vector;

    EXPECT_EQ(0u, const_vector.size());
    EXPECT_TRUE(const_vector.empty());
    EXPECT_TRUE(const_vector.begin() == const_vector.end());
}

// Direct array access.
TYPED_TEST(SmallVectorTest, DirectVectorTest) {
    EXPECT_EQ(0u, this->the_vector.size());
    this->the_vector.reserve(4);
    EXPECT_LE(4u, this->the_vector.capacity());
    EXPECT_EQ(0, Constructable::get_num_constructor_calls());
    this->the_vector.push_back(1);
    this->the_vector.push_back(2);
    this->the_vector.push_back(3);
    this->the_vector.push_back(4);
    EXPECT_EQ(4u, this->the_vector.size());
    EXPECT_EQ(8, Constructable::get_num_constructor_calls());
    EXPECT_EQ(1, this->the_vector[0].get_value());
    EXPECT_EQ(2, this->the_vector[1].get_value());
    EXPECT_EQ(3, this->the_vector[2].get_value());
    EXPECT_EQ(4, this->the_vector[3].get_value());
}

TYPED_TEST(SmallVectorTest, IteratorTest) {
    std::list<int> list;
    this->the_vector.insert(this->the_vector.end(), list.begin(), list.end());
}

template <typename InvalidType>
class DualSmallVectorsTest;

template <typename VectorT1, typename VectorT2>
class DualSmallVectorsTest<std::pair<VectorT1, VectorT2>>
        : public SmallVectorTestBase {
protected:
    VectorT1 the_vector;
    VectorT2 other_vector;

    template <typename T, unsigned N>
    static unsigned num_builtin_elms(const SmallVector<T, N>&) {
        return N;
    }
};

typedef ::testing::Types<
        // Small mode -> Small mode.
        std::pair<SmallVector<Constructable, 4>, SmallVector<Constructable, 4>>,
        // Small mode -> Big mode.
        std::pair<SmallVector<Constructable, 4>, SmallVector<Constructable, 2>>,
        // Big mode -> Small mode.
        std::pair<SmallVector<Constructable, 2>, SmallVector<Constructable, 4>>,
        // Big mode -> Big mode.
        std::pair<SmallVector<Constructable, 2>, SmallVector<Constructable, 2>>>
        DualSmallVectorTestTypes;

TYPED_TEST_CASE(DualSmallVectorsTest, DualSmallVectorTestTypes);

TYPED_TEST(DualSmallVectorsTest, MoveAssignment) {
    SCOPED_TRACE("MoveAssignTest-DualVectorTypes");

    // Set up our vector with four elements.
    for (unsigned it = 0; it < 4; ++it)
        this->other_vector.push_back(Constructable(it));

    const Constructable* orig_data_ptr = this->other_vector.data();

    // Move-assign from the other vector.
    this->the_vector = std::move(
            static_cast<SmallVectorImpl<Constructable>&>(this->other_vector));

    // Make sure we have the right result.
    this->assert_values_in_order(this->the_vector, 4u, 0, 1, 2, 3);

    // Make sure the # of constructor/destructor calls line up. There
    // are two live objects after clearing the other vector.
    this->other_vector.clear();
    EXPECT_EQ(Constructable::get_num_constructor_calls() - 4,
              Constructable::get_num_destructor_calls());

    // If the source vector (other_vector) was in small-mode, assert that we
    // just
    // moved the data pointer over.
    EXPECT_TRUE(this->num_builtin_elms(this->other_vector) == 4 ||
                this->the_vector.data() == orig_data_ptr);

    // There shouldn't be any live objects any more.
    this->the_vector.clear();
    EXPECT_EQ(Constructable::get_num_constructor_calls(),
              Constructable::get_num_destructor_calls());

    // We shouldn't have copied anything in this whole process.
    // EXPECT_EQ(Constructable::get_num_copy_constructor_calls(), 0);
}

struct NotAssignable {
    int& x;
    NotAssignable(int& x) : x(x) {}
};

TEST(SmallVectorCustomTest, NoAssignTest) {
    int x = 0;
    SmallVector<NotAssignable, 2> vec;
    vec.push_back(NotAssignable(x));
    x = 42;
    EXPECT_EQ(42, vec.pop_back_val().x);
}

struct MovedFrom {
    bool has_value;
    MovedFrom() : has_value(true) {}
    MovedFrom(MovedFrom&& m) : has_value(m.has_value) { m.has_value = false; }
    MovedFrom& operator=(MovedFrom&& m) {
        has_value = m.has_value;
        m.has_value = false;
        return *this;
    }
};

TEST(SmallVectorTest, MidInsert) {
    SmallVector<MovedFrom, 3> v;
    v.push_back(MovedFrom());
    v.insert(v.begin(), MovedFrom());
    for (MovedFrom& m : v)
        EXPECT_TRUE(m.has_value);
}

enum EmplaceableArgstate {
    EAS_Defaulted,
    EAS_Arg,
    EAS_LValue,
    EAS_RValue,
    EAS_Failure
};
template <int it>
struct EmplaceableArg {
    EmplaceableArgstate state;
    EmplaceableArg() : state(EAS_Defaulted) {}
    EmplaceableArg(EmplaceableArg&& x)
            : state(x.state == EAS_Arg ? EAS_RValue : EAS_Failure) {}
    EmplaceableArg(EmplaceableArg& x)
            : state(x.state == EAS_Arg ? EAS_LValue : EAS_Failure) {}

    explicit EmplaceableArg(bool) : state(EAS_Arg) {}

private:
    EmplaceableArg& operator=(EmplaceableArg&&) = delete;
    EmplaceableArg& operator=(const EmplaceableArg&) = delete;
};

enum Emplaceablestate { ES_Emplaced, ES_Moved };
struct Emplaceable {
    EmplaceableArg<0> a0;
    EmplaceableArg<1> a1;
    EmplaceableArg<2> a2;
    EmplaceableArg<3> a3;
    Emplaceablestate state;

    Emplaceable() : state(ES_Emplaced) {}

    template <class A0Ty>
    explicit Emplaceable(A0Ty&& a0)
            : a0(std::forward<A0Ty>(a0)), state(ES_Emplaced) {}

    template <class A0Ty, class A1Ty>
    Emplaceable(A0Ty&& a0, A1Ty&& a1)
            : a0(std::forward<A0Ty>(a0)),
              a1(std::forward<A1Ty>(a1)),
              state(ES_Emplaced) {}

    template <class A0Ty, class A1Ty, class A2Ty>
    Emplaceable(A0Ty&& a0, A1Ty&& a1, A2Ty&& a2)
            : a0(std::forward<A0Ty>(a0)),
              a1(std::forward<A1Ty>(a1)),
              a2(std::forward<A2Ty>(a2)),
              state(ES_Emplaced) {}

    template <class A0Ty, class A1Ty, class A2Ty, class A3Ty>
    Emplaceable(A0Ty&& a0, A1Ty&& a1, A2Ty&& a2, A3Ty&& a3)
            : a0(std::forward<A0Ty>(a0)),
              a1(std::forward<A1Ty>(a1)),
              a2(std::forward<A2Ty>(a2)),
              a3(std::forward<A3Ty>(a3)),
              state(ES_Emplaced) {}

    Emplaceable(Emplaceable&&) : state(ES_Moved) {}
    Emplaceable& operator=(Emplaceable&&) {
        state = ES_Moved;
        return *this;
    }

private:
    Emplaceable(const Emplaceable&) = delete;
    Emplaceable& operator=(const Emplaceable&) = delete;
};

TEST(SmallVectorTest, EmplaceBack) {
    EmplaceableArg<0> a0(true);
    EmplaceableArg<1> a1(true);
    EmplaceableArg<2> a2(true);
    EmplaceableArg<3> a3(true);
    {
        SmallVector<Emplaceable, 3> v;
        v.emplace_back();
        EXPECT_TRUE(v.size() == 1);
        EXPECT_TRUE(v.back().state == ES_Emplaced);
        EXPECT_TRUE(v.back().a0.state == EAS_Defaulted);
        EXPECT_TRUE(v.back().a1.state == EAS_Defaulted);
        EXPECT_TRUE(v.back().a2.state == EAS_Defaulted);
        EXPECT_TRUE(v.back().a3.state == EAS_Defaulted);
    }
    {
        SmallVector<Emplaceable, 3> v;
        v.emplace_back(std::move(a0));
        EXPECT_TRUE(v.size() == 1);
        EXPECT_TRUE(v.back().state == ES_Emplaced);
        EXPECT_TRUE(v.back().a0.state == EAS_RValue);
        EXPECT_TRUE(v.back().a1.state == EAS_Defaulted);
        EXPECT_TRUE(v.back().a2.state == EAS_Defaulted);
        EXPECT_TRUE(v.back().a3.state == EAS_Defaulted);
    }
    {
        SmallVector<Emplaceable, 3> v;
        v.emplace_back(a0);
        EXPECT_TRUE(v.size() == 1);
        EXPECT_TRUE(v.back().state == ES_Emplaced);
        EXPECT_TRUE(v.back().a0.state == EAS_LValue);
        EXPECT_TRUE(v.back().a1.state == EAS_Defaulted);
        EXPECT_TRUE(v.back().a2.state == EAS_Defaulted);
        EXPECT_TRUE(v.back().a3.state == EAS_Defaulted);
    }
    {
        SmallVector<Emplaceable, 3> v;
        v.emplace_back(a0, a1);
        EXPECT_TRUE(v.size() == 1);
        EXPECT_TRUE(v.back().state == ES_Emplaced);
        EXPECT_TRUE(v.back().a0.state == EAS_LValue);
        EXPECT_TRUE(v.back().a1.state == EAS_LValue);
        EXPECT_TRUE(v.back().a2.state == EAS_Defaulted);
        EXPECT_TRUE(v.back().a3.state == EAS_Defaulted);
    }
    {
        SmallVector<Emplaceable, 3> v;
        v.emplace_back(std::move(a0), std::move(a1));
        EXPECT_TRUE(v.size() == 1);
        EXPECT_TRUE(v.back().state == ES_Emplaced);
        EXPECT_TRUE(v.back().a0.state == EAS_RValue);
        EXPECT_TRUE(v.back().a1.state == EAS_RValue);
        EXPECT_TRUE(v.back().a2.state == EAS_Defaulted);
        EXPECT_TRUE(v.back().a3.state == EAS_Defaulted);
    }
    {
        SmallVector<Emplaceable, 3> v;
        v.emplace_back(std::move(a0), a1, std::move(a2), a3);
        EXPECT_TRUE(v.size() == 1);
        EXPECT_TRUE(v.back().state == ES_Emplaced);
        EXPECT_TRUE(v.back().a0.state == EAS_RValue);
        EXPECT_TRUE(v.back().a1.state == EAS_LValue);
        EXPECT_TRUE(v.back().a2.state == EAS_RValue);
        EXPECT_TRUE(v.back().a3.state == EAS_LValue);
    }
    {
        SmallVector<int, 1> v;
        v.emplace_back();
        v.emplace_back(42);
        EXPECT_EQ(2U, v.size());
        EXPECT_EQ(0, v[0]);
        EXPECT_EQ(42, v[1]);
    }
}

TEST(SmallVectorTest, FindTest) {
    SmallVector<int> v = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(find(v, 3), &v[3]);
    EXPECT_EQ(find(v, 6), &v[6]);
    v[3] = 6;
    EXPECT_EQ(find(v, 6), &v[3]);
    EXPECT_EQ(find(v, 2), &v[2]);
}

TEST(SmallVectorTest, InitializerList) {
    SmallVector<int, 2> v1 = {};
    EXPECT_TRUE(v1.empty());
    v1 = {0, 0};
    EXPECT_TRUE(std::equal(v1.begin(), v1.end(),
                           std::initializer_list<int>({0, 0}).begin()));
    v1 = {-1, -1};
    EXPECT_TRUE(std::equal(v1.begin(), v1.end(),
                           std::initializer_list<int>({-1, -1}).begin()));

    SmallVector<int, 2> v2 = {1, 2, 3, 4};
    EXPECT_TRUE(std::equal(v2.begin(), v2.end(),
                           std::initializer_list<int>({1, 2, 3, 4}).begin()));
    v2.assign({4});
    EXPECT_TRUE(std::equal(v2.begin(), v2.end(),
                           std::initializer_list<int>({4}).begin()));
    v2.append({3, 2});
    EXPECT_TRUE(std::equal(v2.begin(), v2.end(),
                           std::initializer_list<int>({4, 3, 2}).begin()));
    v2.insert(v2.begin() + 1, 5);
    EXPECT_TRUE(std::equal(v2.begin(), v2.end(),
                           std::initializer_list<int>({4, 5, 3, 2}).begin()));
}

TEST(SmallVectorTest, PutElementWithinVectorIntoItself) {
    SmallVector<Constructable> vector;
    vector.emplace_back(0);
    for (size_t i = 0; i < 10; ++i) {
        vector.push_back(vector[0]);
    }
    vector.assign(30, vector[0]);
    vector.resize(90, vector[0]);
    vector.append(270, vector[0]);
    for (size_t i = 0; i < 1000; ++i) {
        vector.insert(&vector[0], vector[0]);
    }
    vector.insert(vector.begin(), 3000, vector[0]);
}

TEST(SmallVectorTest, SwapMoveOnly) {
    auto run = [](size_t nr0, size_t nr1, bool use_std_swap) {
        SmallVector<MoveOnly, 2> vec0, vec1;
        for (size_t i = 0; i < nr0; ++i) {
            vec0.emplace_back(i * 2 + 1);
        }
        for (size_t i = 0; i < nr1; ++i) {
            vec1.emplace_back(i * 2 + 2);
        }
        if (use_std_swap) {
            std::swap(vec0, vec1);
        } else {
            vec0.swap(vec1);
        }
        ASSERT_EQ(nr0, vec1.size());
        ASSERT_EQ(nr1, vec0.size());
        for (size_t i = 0; i < nr0; ++i) {
            ASSERT_EQ(static_cast<int>(i * 2 + 1), vec1[i].val);
        }
        for (size_t i = 0; i < nr1; ++i) {
            ASSERT_EQ(static_cast<int>(i * 2 + 2), vec0[i].val);
        }
    };
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            run(i, j, 0);
            run(i, j, 1);
        }
    }
}

}  // anonymous namespace

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
