#ifndef EVENT_HH
#define EVENT_HH

#include "G4UserEventAction.hh"
#include "G4Event.hh"

#include "G4AnalysisManager.hh"

#include "run.hh"
#include "G4UnitsTable.hh"

class MyEventAction : public G4UserEventAction
{
public:
    MyEventAction(MyRunAction*);
    ~MyEventAction();
    
    virtual void BeginOfEventAction(const G4Event*);
    virtual void EndOfEventAction(const G4Event*);

    void AddEdep(G4double edep)
    {
        fEdep += edep;
    }

    // void countParticles()
    // {
    //     count++;
    // }

private:
    // G4int count;
    G4double fEdep;
};

#endif