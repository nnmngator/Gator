#include "GatorPolar.h"



GatorPolar::GatorPolar(float A, float phi = 0.f) : 
	A(A), phi(phi)
{}


GatorPolar::~GatorPolar()
{}

GatorPolar GatorPolar::operator*(const GatorPolar & o)
{
	return GatorPolar(A * o.A, phi + o.phi);
}
