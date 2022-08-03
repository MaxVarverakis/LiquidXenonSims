#ifndef EVENT_HH
#define EVENT_HH

#include "G4UserEventAction.hh"
#include "G4Event.hh"

#include "G4AnalysisManager.hh"

#include "G4UnitsTable.hh"
#include "G4SDManager.hh"
#include "G4HCofThisEvent.hh"
#include "G4VHitsCollection.hh"

#include "run.hh"
#include "target.hh"
#include "window.hh"
#include "targetHit.hh"
#include "windowHit.hh"

class MyEventAction : public G4UserEventAction
{
public:
    MyEventAction(MyRunAction*);
    ~MyEventAction() override;
    
    void BeginOfEventAction(const G4Event*) override;
    void EndOfEventAction(const G4Event*) override;

    // void AddEdep(G4double edep)
    // {
    //     fTargetEdep += edep;
    // }

    // void countParticles()
    // {
    //     count++;
    // }

private:
    // std::array<G4int, 1> fTargetHCID = { -1 };
    // std::array<G4int, 2> fWindowInHCID = { -1 };
    // std::array<G4int, 2> fWindowOutHCID = { -1};

    G4int fTargetHCID {-1};
    G4int fWindowInHCID {-1};
    G4int fWindowOutHCID{-1};

    // G4int count;
    // G4double fTargetEdep;
};

#endif