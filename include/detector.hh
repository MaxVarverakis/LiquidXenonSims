#ifndef DETECTOR_HH
#define DETECTOR_HH

#include <iostream>
#include <fstream>
#include <math.h>
#include "G4VSensitiveDetector.hh"

#include "G4SystemOfUnits.hh"
#include "G4AnalysisManager.hh"
#include "G4RunManager.hh"
#include "G4StepStatus.hh"
#include "G4UnitsTable.hh"

class MySensitiveDetector : public G4VSensitiveDetector
{
public:
    MySensitiveDetector(G4String name);
    ~MySensitiveDetector();
    
    
private:
    virtual G4bool ProcessHits(G4Step*, G4TouchableHistory*);
    // int count;
    // G4double Etot;
};

#endif