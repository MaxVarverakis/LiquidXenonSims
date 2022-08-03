#ifndef WINDOWHIT_HH
#define WINDOWHIT_HH

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4Allocator.hh"

class G4AttDef;
class G4AttValue;

class WindowHit : public G4VHit
{
public:
    public:
    WindowHit();
    // WindowHit(G4int cellID);
    WindowHit(const WindowHit &right) = default;
    ~WindowHit() override;

    // WindowHit& operator=(const WindowHit &right) = default;
    // G4bool operator==(const WindowHit &right) const;

    inline void *operator new(size_t);
    inline void operator delete(void *aHit);

    // void Draw() override;
    // const std::map<G4String,G4AttDef>* GetAttDefs() const override;
    // std::vector<G4AttValue>* CreateAttValues() const override;
    // void Print() override;

    // void SetCellID(G4int z) { fCellID = z; }
    // G4int GetCellID() const { return fCellID; }

    void SetEdep(G4double de) { fEdep = de; }
    void AddEdep(G4double de) { fEdep += de; }
    G4double GetEdep() const { return fEdep; }

private:
    // G4int fCellID = -1;
    G4double fEdep = 0.;
};

using WindowHitsCollection = G4THitsCollection<WindowHit>;

extern G4ThreadLocal G4Allocator<WindowHit>* WindowHitAllocator;

inline void* WindowHit::operator new(size_t)
{
    if (!WindowHitAllocator) 
    {
        WindowHitAllocator = new G4Allocator<WindowHit>;
    }
    return (void*)WindowHitAllocator->MallocSingle();
}

inline void WindowHit::operator delete(void* aHit)
{
    WindowHitAllocator -> FreeSingle((WindowHit*) aHit);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif