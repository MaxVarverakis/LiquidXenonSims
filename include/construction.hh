#ifndef CONSTRUCTION_HH
#define CONSTRUCTION_HH

#include "G4SystemOfUnits.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalvolume.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4GenericMessenger.hh"
#include "G4PhysicalConstants.hh"

#include "target.hh"
#include "window.hh"

class MyDetectorConstruction : public G4VUserDetectorConstruction
{
public:
    MyDetectorConstruction();
    ~MyDetectorConstruction() override;
    
    // G4LogicalVolume *GetScoringVolume() const { return fScoringVolume; }

    G4VPhysicalVolume  *Construct() override;
    void ConstructTarget();
    void ConstructWindows(G4double targetWidth);
    void ConstructSDandField() override;

private:
    G4Tubs *solidWindowIn, *solidWindowOut;
    G4Box *solidWorld, *solidTarget;
    G4LogicalVolume *logicWindowIn, *logicWindowOut, *logicTarget, *logicWorld;
    G4VPhysicalVolume *physWorld, *physTarget;
    G4GenericMessenger *target;

    G4bool liquidXenon;
    G4bool windows = true; // throws error if false

    // G4Element *elXe;
    G4Material *air, *windowMaterial, *targetMaterial;
    // , *lqdXe

    G4double L_RL, n, dLRL;
    G4ThreeVector targetPos;
};

#endif