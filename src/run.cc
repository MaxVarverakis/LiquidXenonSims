#include "run.hh"

MyRunAction::MyRunAction()
{
    G4AnalysisManager *man = G4AnalysisManager::Instance();
    
    // man -> CreateNtuple("e-", "Hits");
    // man -> CreateNtupleDColumn("fEnergy");
    // man -> CreateNtupleDColumn("fTraverseWidth");
    // man -> CreateNtupleDColumn("fAngle");
    // man -> FinishNtuple(0);
    
    man -> CreateNtuple("e+", "Hits");
    man -> CreateNtupleDColumn("fEnergy");
    man -> CreateNtupleDColumn("fTraverseWidth");
    man -> CreateNtupleDColumn("fAngle");
    man -> FinishNtuple(0);

    // Energy Deposition
    man -> CreateNtuple("Data", "Edep");
    man -> CreateNtupleDColumn("fEdep");
    man -> FinishNtuple(1);
}

MyRunAction::~MyRunAction()
{}

void MyRunAction::BeginOfRunAction(const G4Run* run)
{
    G4AnalysisManager *man = G4AnalysisManager::Instance();
    
    G4int runID = run -> GetRunID();

    std::stringstream strRunID;
    strRunID << runID;

    man -> OpenFile("out" + strRunID.str() + ".csv");

    G4cout << runID << G4endl;
}

void MyRunAction::EndOfRunAction(const G4Run*)
{
    G4AnalysisManager *man = G4AnalysisManager::Instance();

    man -> Write();
    man -> CloseFile();
    
}