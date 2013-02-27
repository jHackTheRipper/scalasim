#ifndef ALGORITHM_H
#define ALGORITHM_H

/** Adaptation of std::copy_if only available in C++11
    This is full of s**t, not generic at all, and might make you blind
*/
namespace std {
    template <class InputIterator, class OutputIterator, class OtherIterator, class UnaryPredicate>
      OutputIterator my_copy_if (InputIterator first, InputIterator last,
                                 OtherIterator firstTopLevel, OtherIterator lastTopLevel,
                                 OutputIterator result, UnaryPredicate pred)
    {
      while (first!=last) {
        if (pred(*first)) {
          *result = std::make_pair<int, int>(lastTopLevel - firstTopLevel, last - first);
          ++result;
        }
        ++first;
      }
      return result;
    }
}

#endif // ALGORITHM_H
