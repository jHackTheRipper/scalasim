#ifndef ZIP_H
#define ZIP_H

/** Equivalent of the zip function from Scala
 *  Adapted from http://stackoverflow.com/a/8511125/470341 */
template <typename C1, typename C2>
class zip_container {
    C1* c1;
    C2* c2;

public:

    // CUDA does not allow C++11 constructs yet
    // thus had to get rid of decltype
    typedef typename C1::iterator tuple_1;
    typedef typename C2::iterator tuple_2;

    typedef boost::tuple<
        tuple_1,
        tuple_2
    > tuple;

    zip_container(C1& c1, C2& c2) : c1(&c1), c2(&c2) {}

    typedef typename boost::zip_iterator<tuple> iterator;

    iterator begin() const { return boost::make_zip_iterator(boost::make_tuple(c1->begin(), c2->begin() )); }
    iterator end()   const { return boost::make_zip_iterator(boost::make_tuple(c1->end(),   c2->end()   ));   }

    inline std::size_t size() { return end() - begin(); }
};

template <typename C1, typename C2>
zip_container<C1, C2> zip(C1& c1, C2& c2) {
    return zip_container<C1, C2>(c1, c2);
}

#endif // ZIP_H
