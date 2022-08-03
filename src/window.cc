#include "window.hh"

SensitiveWindow::SensitiveWindow(G4String name) : G4VSensitiveDetector(name)
{
    collectionName.insert( "windowCollection");
    // fEdep = 0.;
}

SensitiveWindow::~SensitiveWindow()
{}

void SensitiveWindow::Initialize(G4HCofThisEvent* hce)
{
    fHitsCollection = new WindowHitsCollection(SensitiveDetectorName, collectionName[0]);
    if (fHCID < 0)
    { fHCID = G4SDManager::GetSDMpointer() -> GetCollectionID(fHitsCollection); }
    hce -> AddHitsCollection(fHCID, fHitsCollection);

    // fill window hits with zero energy deposition
    fHitsCollection -> insert(new WindowHit());
}

G4bool SensitiveWindow::ProcessHits(G4Step *step, G4TouchableHistory *R0history)
{
    // G4String particleName = step -> GetTrack() -> GetDefinition() -> GetParticleName();

    G4StepPoint *preStepPoint = step -> GetPreStepPoint();
    // G4StepPoint *postStepPoint = step -> GetPostStepPoint();
    
    auto copyNo = preStepPoint -> GetTouchable() -> GetVolume() -> GetCopyNo();
    
    auto hit = (*fHitsCollection)[copyNo];
    hit -> AddEdep(step -> GetTotalEnergyDeposit());
    
    // AddEdep(step -> GetTotalEnergyDeposit());
    // G4Track *track = step -> GetTrack();
    // track -> SetTrackStatus(fStopAndKill);

    // G4String particleName = track -> GetDefinition() -> GetParticleName();

    // G4StepPoint *preStepPoint = step -> GetPreStepPoint();
    // G4StepPoint *postStepPoint = step -> GetPostStepPoint();
    
    // G4AnalysisManager *man = G4AnalysisManager::Instance();
    return true;
}
