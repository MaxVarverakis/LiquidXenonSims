#ifndef WINDOW_HH
#define WINDOW_HH

#include <iostream>
#include "G4VSensitiveDetector.hh"

#include "G4SystemOfUnits.hh"
#include "G4AnalysisManager.hh"
#include "G4RunManager.hh"
#include "G4StepStatus.hh"
#include "G4UnitsTable.hh"
#include "G4HCofThisEvent.hh"
#include "G4THitsCollection.hh"
#include "G4SDManager.hh"

#include "windowHit.hh"

class G4Step;
class G4HCofThisEvent;
class G4TouchableHistory;

class SensitiveWindow : public G4VSensitiveDetector
{
public:
    SensitiveWindow(G4String name);
    ~SensitiveWindow() override;
    
    // void SetEdep(G4double de) { fEdep = de; }
    // void AddEdep(G4double de) { fEdep += de; }
    // G4double GetEdep() const { return fEdep; }

    void Initialize(G4HCofThisEvent *HCE) override;
    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
    
private:
    // G4double fEdep;
    // G4int fWindowID;

    WindowHitsCollection* fHitsCollection = nullptr;
    G4int fHCID = -1;
};

#endif