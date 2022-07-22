#include "event.hh"

MyEventAction::MyEventAction(MyRunAction* runAction)
{
    // count = 0;
    fEdep = 0.;
}

MyEventAction::~MyEventAction()
{}

void MyEventAction::BeginOfEventAction(const G4Event*)
{
    // count = 0;
    fEdep = 0.;
}

void MyEventAction::EndOfEventAction(const G4Event*)
{
    G4AnalysisManager *man = G4AnalysisManager::Instance();

    // G4cout << "Deposition: " << fEdep << G4endl;
    // G4cout << "Deposition: " << G4BestUnit(fEdep, "Energy") << G4endl;

    man -> FillNtupleDColumn(1, 0, fEdep);
    man -> AddNtupleRow(1);
    // man -> FillNtupleIColumn(1, 0, count);
    // man -> AddNtupleRow(1);
}