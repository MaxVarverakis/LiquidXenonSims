#include "event.hh"

using std::array;

namespace {

// Utility function which finds a hit collection with the given Id
// and print warnings if not found
G4VHitsCollection* GetHC(const G4Event* event, G4int collId) {
    auto hce = event -> GetHCofThisEvent();
    if (!hce) {
        G4ExceptionDescription msg;
        msg << "No hits collection of this event found." << G4endl;
        G4Exception("EventAction::EndOfEventAction()",
                    "Code001", JustWarning, msg);
        return nullptr;
    }

    auto hc = hce -> GetHC(collId);
    if ( ! hc) {
        G4ExceptionDescription msg;
        msg << "Hits collection " << collId << " of this event not found." << G4endl;
        G4Exception("EventAction::EndOfEventAction()",
                    "Code001", JustWarning, msg);
    }
    return hc;
}

}

MyEventAction::MyEventAction(MyRunAction* runAction)
{}

MyEventAction::~MyEventAction()
{}

void MyEventAction::BeginOfEventAction(const G4Event*)
{
    // Find hit collections and histogram Ids by names (just once)
    // and save them in the data members of this class

    if (fTargetHCID == -1) 
    {
        auto sdManager = G4SDManager::GetSDMpointer();
        // auto analysisManager = G4AnalysisManager::Instance();

        // hits collections names
        // array<G4String, 1> tHCName = {{ "target/targetCollection" }};
        // array<G4String, 2> wHCName = {{ "BeIn/windowCollection", "BeOut/windowCollection" }};

        // hit collections IDs
        fTargetHCID = sdManager -> GetCollectionID("target/targetCollection");
        fWindowInHCID = sdManager -> GetCollectionID("BeIn/windowCollection");
        fWindowOutHCID = sdManager -> GetCollectionID("BeOut/windowCollection");
    }
}

void MyEventAction::EndOfEventAction(const G4Event* event)
{
    G4AnalysisManager *man = G4AnalysisManager::Instance();
    
    G4cout << "-------------------\n\tWindow Hits\t\n-------------------" << G4endl;

    // first window hits
    auto ihc = GetHC(event, fWindowInHCID);
    if ( ! ihc ) return;

    G4double fWindowInEdep = 0.;
    for (unsigned int i = 0; i < ihc -> GetSize(); ++i)
    {
        auto ihit = static_cast<WindowHit*>(ihc -> GetHit(i));
        fWindowInEdep = ihit -> GetEdep();
    }
    man -> FillNtupleDColumn(1, 1, fWindowInEdep);
    G4cout << fWindowInEdep << G4endl;

    // target hits
    auto thc = GetHC(event, fTargetHCID);
    if ( ! thc ) return;

    G4double fTargetEdep = 0.;
    for (unsigned int i = 0; i < thc -> GetSize(); ++i)
    {
        auto thit = static_cast<TargetHit*>(thc -> GetHit(i));
        fTargetEdep = thit -> GetEdep();
    }
    man -> FillNtupleDColumn(1, 0, fTargetEdep);
    G4cout << fTargetEdep << G4endl;

    // last window hits
    auto ohc = GetHC(event, fWindowOutHCID);
    if ( ! ohc ) return;

    G4double fWindowOutEdep = 0.;
    for (unsigned int i = 0; i < ohc -> GetSize(); ++i)
    {
        auto ohit = static_cast<WindowHit*>(ohc -> GetHit(i));
        fWindowOutEdep = ohit -> GetEdep();
    }
    man -> FillNtupleDColumn(1, 2, fWindowOutEdep);
    G4cout << fWindowOutEdep << G4endl;

    man -> AddNtupleRow(1);
}