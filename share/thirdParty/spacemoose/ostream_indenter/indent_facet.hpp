#pragma once

#include <locale>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cassert>



class indent_facet : public std::codecvt<char, char, std::mbstate_t> {
public:
	explicit indent_facet( int indent_level, size_t ref = 0)
		: std::codecvt<char, char, std::mbstate_t>(ref), m_indentation_level(indent_level) {}
	typedef std::codecvt_base::result result;
	typedef std::codecvt<char, char, std::mbstate_t> parent;
	typedef parent::intern_type intern_type;
	typedef parent::extern_type extern_type;
	typedef parent::state_type  state_type;

	int &state(state_type &s) const { return *reinterpret_cast<int *>(&s); }

protected:
	virtual result do_out(state_type &need_indentation,
		const intern_type *from, const intern_type *from_end, const intern_type *&from_next,
		extern_type *to, extern_type *to_end, extern_type *&to_next
		) const override;

	// Override so the do_out() virtual function is called.
	virtual bool do_always_noconv() const throw() override {
		return m_indentation_level==0;
	}
	int m_indentation_level = 0;

};

inline indent_facet::result indent_facet::do_out(state_type &need_indentation,
	const intern_type *from, const intern_type *from_end, const intern_type *&from_next,
	extern_type *to, extern_type *to_end, extern_type *&to_next
	) const
{
	result res = std::codecvt_base::noconv;
	for (; (from < from_end) && (to < to_end); ++from, ++to) {
		// 0 indicates that the last character seen was a newline.
		// thus we will print a tab before it. Ignore it the next
		// character is also a newline
		if ((state(need_indentation) == 0) && (*from != '\n')) {
			res = std::codecvt_base::ok;
			state(need_indentation) = 1;
			for(int i=0; i<m_indentation_level; ++i){
				*to = '\t'; ++to;
			}
			if (to == to_end) {
				res = std::codecvt_base::partial;
				break;
			}
		}
		*to = *from; // Copy the next character.

		// If the character copied was a '\n' mark that state
		if (*from == '\n') {
			state(need_indentation) = 0;
		}
	}

	if (from != from_end) {
		res = std::codecvt_base::partial;
	}
	from_next = from;
	to_next = to;

	return res;
};



/// I hate the way I solved this, but I can't think of a better way
/// around the problem.  I even asked stackoverflow for help:
///
///   http://stackoverflow.com/questions/32480237/apply-a-facet-to-all-stream-output-use-custom-string-manipulators
///
///
namespace  indent_manip{

static const int index = std::ios_base::xalloc();

inline static std::ostream & push(std::ostream& os)
{
	auto ilevel = ++os.iword(index);
	os.imbue(std::locale(os.getloc(), new indent_facet(ilevel)));
	return os;
}

inline std::ostream& pop(std::ostream& os)
{
	auto ilevel = (os.iword(index)>0) ? --os.iword(index) : 0;
	os.imbue(std::locale(os.getloc(), new indent_facet(ilevel)));
	return os;
}

/// Clears the ostream indentation set, but NOT the raii_guard.
inline std::ostream& clear(std::ostream& os)
{
	os.iword(index) = 0;
	os.imbue(std::locale(os.getloc(), new indent_facet(0)));
	return os;
}



/// Provides a RAII guard around your manipulation.
class raii_guard
{
public:
	raii_guard(std::ostream& os):
		oref(os),
		start_level(os.iword(index))
	{}

	~raii_guard()
	{
		reset();
	}

	/// Resets the streams indentation level to the point itw as at
	/// when the guard was created.
	void reset()
	{
		oref.iword(index) = start_level;
		oref.imbue(std::locale(oref.getloc(), new indent_facet(start_level)));
	}

private:
	std::ostream& oref;
	int start_level;
};

}
