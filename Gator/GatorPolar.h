#pragma once
class GatorPolar
{
	float A, phi;

public:
	GatorPolar(float A, float phi = 0.f);
	~GatorPolar();

	//GatorPolar operator+(const GatorPolar& o);
	GatorPolar operator*(const GatorPolar& o);
};

