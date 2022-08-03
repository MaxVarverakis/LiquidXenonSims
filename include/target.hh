#ifndef TARGET_HH
#define TARGET_HH

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

#include "targetHit.hh"

class G4Step;
class G4HCofThisEvent;
class G4TouchableHistory;

class SensitiveTarget : public G4VSensitiveDetector
{
public:
    SensitiveTarget(G4String name);
    ~SensitiveTarget() override;
    
    // void SetEdep(G4double de) { fEdep = de; }
    // void AddEdep(G4double de) { fEdep += de; }
    // G4double GetEdep() const { return fEdep; }

    void Initialize(G4HCofThisEvent *HCE) override;
    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
    
private:
    // G4double fEdep;
    // G4int fTargetID;

    TargetHitsCollection* fHitsCollection = nullptr;
    G4int fHCID = -1;
};

#endif