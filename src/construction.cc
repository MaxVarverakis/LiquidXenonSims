#include "construction.hh"

MyDetectorConstruction::MyDetectorConstruction()
{
    fMessenger = new G4GenericMessenger(this, "/detector/", "Detector Construction");

    fMessenger -> DeclareProperty("radLengths", n, "Number of radiation lengths");
    
    L_RL = 0.4094 * cm;
    // L_RL = 2.872 * cm;

    n = 0.250;
}

MyDetectorConstruction::~MyDetectorConstruction()
{}

G4VPhysicalVolume *MyDetectorConstruction::Construct()
{
    G4NistManager *nist = G4NistManager::Instance();

    // liquid Xenon info: https://pdg.lbl.gov/2019/AtomicNuclearProperties/HTML/liquid_xenon_Xe.html
    G4double lqdXe_density = 2.953 * g/cm3;
    G4double A = 131.2930 * g/mole;  // atomic mass
    G4double Z = 54.;  // atomic number

    elXe = new G4Element("Xenon", "Xe", Z, A);
    lqdXe = new G4Material("LiquidXe", lqdXe_density, 1);
    lqdXe -> AddElement(elXe, 1.0);

    // G4Material *lXe = nist -> FindOrBuildMaterial("G4_lXe");
    G4Material *air = nist -> FindOrBuildMaterial("G4_Galactic");
    G4Material *targetMaterial = nist -> FindOrBuildMaterial("G4_Ta");
    // G4Material *targetMaterial = lqdXe;

    G4Box *solidWorld = new G4Box("solidWorld", 0.5 * m, 0.5 * m, 0.5 * m);
    G4LogicalVolume *logicWorld = new G4LogicalVolume(solidWorld, air, "logicWorld");

    G4VPhysicalVolume *physWorld = new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), logicWorld, "physWorld", 0, false, 0, true);


    G4Box *solidTarget = new G4Box("solidTarget", 0.25 * m, 0.25 * m, n * L_RL);
    logicTarget = new G4LogicalVolume(solidTarget, targetMaterial, "logicTarget");
    new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), logicTarget, "physTarget", logicWorld, false, 0, true);

    fScoringVolume = logicTarget;

    return physWorld;
}

void MyDetectorConstruction::ConstructSDandField()
{
    MySensitiveDetector *sensDet = new MySensitiveDetector("SensitiveDetector");

    logicTarget -> SetSensitiveDetector(sensDet);
}