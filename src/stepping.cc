#include "stepping.hh"

MySteppingAction::MySteppingAction(MyEventAction *eventAction)
{
    fEventAction = eventAction;
}

MySteppingAction::~MySteppingAction()
{}

void MySteppingAction::UserSteppingAction(const G4Step *step)
{

    // Energy Deposition of incident electron
    G4LogicalVolume *volume = step -> GetPreStepPoint() -> GetTouchableHandle() -> GetVolume() -> GetLogicalVolume();

    const MyDetectorConstruction *detectorConstruction = static_cast <const MyDetectorConstruction*> (G4RunManager::GetRunManager() -> GetUserDetectorConstruction());
    
    G4LogicalVolume *fScoringVolume = detectorConstruction -> GetScoringVolume();
    
    // if (preVolume == fScoringVolume)
    // {
    //     G4double edep = step -> GetTotalEnergyDeposit();
    //     // fEventAction -> AddEdep(edep);
    //     return;
    // }
    // else if (postVolume == fScoringVolume)
    // {
    //     G4double edep = step -> GetTotalEnergyDeposit();
    //     fEventAction -> AddEdep(edep);
    // }
    // else
    // {
    //     return;
    // }

    if(volume != fScoringVolume)
        return;
    
    G4double edep = step -> GetTotalEnergyDeposit();
    fEventAction -> AddEdep(edep);
    // fEventAction -> countParticles();
}
