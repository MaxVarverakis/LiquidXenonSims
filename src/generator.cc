#include "generator.hh"

MyPrimaryGenerator::MyPrimaryGenerator()
{
    fMessenger = new G4GenericMessenger(this, "/detector/", "Detector Construction");
    fMessenger -> DeclareProperty("beam", isBeamDist, "Use beam distribution");

    if (isBeamDist)
    {
        // if using beam distribution, must use macro on startup to initialize beam info! (e.g., beam.mac)
        fBeamGun = new G4GeneralParticleSource();

    }
    else
    {
        fParticleGun = new G4ParticleGun(1);
        G4ParticleDefinition *particleTable = G4ParticleTable::GetParticleTable() -> FindParticle("e-");
        
        fParticleGun -> SetParticleDefinition(particleTable);
        fParticleGun -> SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));
        fParticleGun -> SetParticleEnergy(3.0 * GeV);
        fParticleGun -> SetParticlePosition(G4ThreeVector(0., 0., -0.5 * m));
        
    }

}

MyPrimaryGenerator::~MyPrimaryGenerator()
{
    if (isBeamDist)
    {
        delete fBeamGun;
    }
    else
    {
        delete fParticleGun;
    }
}

void MyPrimaryGenerator::GeneratePrimaries(G4Event *anEvent)
{
    if (isBeamDist)
    {
        fBeamGun -> GeneratePrimaryVertex(anEvent);
    }
    else
    {
        fParticleGun -> GeneratePrimaryVertex(anEvent);
    }
}