"""
TransitKit v3.0 - Universal Target Resolver

The magic: Enter ANY identifier and get ALL cross-matched data.
Supports: Planet names, TIC, KIC, KOI, TOI, EPIC, HD, HIP, 2MASS, Gaia DR3, etc.

Example:
    >>> target = UniversalTarget("WASP-39 b")
    >>> target = UniversalTarget("TIC 374829238")
    >>> target = UniversalTarget("TOI-700 d")
    >>> target = UniversalTarget("Kepler-442b")
"""

import re
import json
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union
from enum import Enum, auto
from functools import lru_cache
import numpy as np


# Lazy imports for optional dependencies
def _import_astroquery():
    from astroquery.simbad import Simbad
    from astroquery.mast import Catalogs, Observations
    from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive as NEA2

    return Simbad, Catalogs, Observations, NasaExoplanetArchive


def _import_astropy():
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astropy.table import Table

    return SkyCoord, u, Table


class TargetType(Enum):
    """Classification of target identifier types."""

    PLANET_NAME = auto()  # WASP-39 b, HD 209458 b
    TIC = auto()  # TESS Input Catalog
    KIC = auto()  # Kepler Input Catalog
    KOI = auto()  # Kepler Object of Interest
    TOI = auto()  # TESS Object of Interest
    EPIC = auto()  # K2 Ecliptic Plane Input Catalog
    HD = auto()  # Henry Draper Catalog
    HIP = auto()  # Hipparcos
    GAIA_DR3 = auto()  # Gaia DR3
    TWO_MASS = auto()  # 2MASS
    COORDINATES = auto()  # RA/Dec
    UNKNOWN = auto()


@dataclass
class StellarParameters:
    """Host star parameters from multiple catalogs."""

    ra: Optional[float] = None
    dec: Optional[float] = None
    teff: Optional[float] = None
    teff_err: Optional[float] = None
    logg: Optional[float] = None
    logg_err: Optional[float] = None
    feh: Optional[float] = None
    feh_err: Optional[float] = None
    radius: Optional[float] = None  # R_sun
    radius_err: Optional[float] = None
    mass: Optional[float] = None  # M_sun
    mass_err: Optional[float] = None
    luminosity: Optional[float] = None  # L_sun
    distance: Optional[float] = None  # pc
    distance_err: Optional[float] = None
    vmag: Optional[float] = None
    tmag: Optional[float] = None
    kepmag: Optional[float] = None
    spectral_type: Optional[str] = None
    source: Optional[str] = None


@dataclass
class PlanetParameters:
    """Planetary parameters from NASA Exoplanet Archive."""

    name: str = ""
    period: Optional[float] = None  # days
    period_err: Optional[float] = None
    t0: Optional[float] = None  # BJD
    t0_err: Optional[float] = None
    duration: Optional[float] = None  # hours
    duration_err: Optional[float] = None
    depth: Optional[float] = None  # ppm or fraction
    depth_err: Optional[float] = None
    rp_rs: Optional[float] = None  # Rp/R*
    rp_rs_err: Optional[float] = None
    a_rs: Optional[float] = None  # a/R*
    a_rs_err: Optional[float] = None
    inc: Optional[float] = None  # degrees
    inc_err: Optional[float] = None
    ecc: Optional[float] = None
    ecc_err: Optional[float] = None
    omega: Optional[float] = None  # deg
    radius: Optional[float] = None  # R_earth
    radius_err: Optional[float] = None
    mass: Optional[float] = None  # M_earth
    mass_err: Optional[float] = None
    teq: Optional[float] = None  # K
    insol: Optional[float] = None  # S_earth
    discovery_method: Optional[str] = None
    discovery_facility: Optional[str] = None
    discovery_year: Optional[int] = None
    confirmed: bool = False
    source: Optional[str] = None


@dataclass
class CrossMatchedIDs:
    """All known identifiers for a target."""

    primary_name: str = ""
    planet_name: Optional[str] = None
    host_name: Optional[str] = None
    tic: Optional[int] = None
    kic: Optional[int] = None
    koi: Optional[str] = None
    toi: Optional[str] = None
    epic: Optional[int] = None
    hd: Optional[str] = None
    hip: Optional[int] = None
    gaia_dr3: Optional[str] = None
    two_mass: Optional[str] = None
    aliases: List[str] = field(default_factory=list)


@dataclass
class AvailableData:
    """Summary of available mission data."""

    tess_sectors: List[int] = field(default_factory=list)
    tess_cadences: List[str] = field(default_factory=list)  # '20s', '2min', 'FFI'
    kepler_quarters: List[int] = field(default_factory=list)
    k2_campaigns: List[int] = field(default_factory=list)
    jwst_programs: List[str] = field(default_factory=list)
    jwst_instruments: List[str] = field(default_factory=list)
    has_transmission_spectrum: bool = False
    has_emission_spectrum: bool = False
    ground_based: List[str] = field(default_factory=list)


class IdentifierParser:
    """Parse and classify any exoplanet/star identifier."""

    # Regex patterns for different identifier types
    PATTERNS = {
        TargetType.TIC: [
            r"^TIC[\s\-_]?(\d+)$",
            r"^TESS[\s\-_]?(\d+)$",
        ],
        TargetType.KIC: [
            r"^KIC[\s\-_]?(\d+)$",
            r"^Kepler[\s\-_]?(\d+)$",
        ],
        TargetType.KOI: [
            r"^KOI[\s\-_]?([\d\.]+)$",
            r"^K(\d+\.\d+)$",
        ],
        TargetType.TOI: [
            r"^TOI[\s\-_]?([\d\.]+)$",
        ],
        TargetType.EPIC: [
            r"^EPIC[\s\-_]?(\d+)$",
            r"^K2[\s\-_]?(\d+)$",
        ],
        TargetType.HD: [
            r"^HD[\s\-_]?(\d+[A-Za-z]?)$",
        ],
        TargetType.HIP: [
            r"^HIP[\s\-_]?(\d+)$",
        ],
        TargetType.GAIA_DR3: [
            r"^Gaia[\s\-_]?DR3[\s\-_]?(\d+)$",
            r"^DR3[\s\-_]?(\d+)$",
        ],
        TargetType.TWO_MASS: [
            r"^2MASS[\s\-_]?J?([\d\+\-]+)$",
        ],
        TargetType.PLANET_NAME: [
            # Named planets: WASP-39 b, HD 209458 b, GJ 1214 b
            r"^([A-Za-z]+[\s\-]?\d+[\s\-]?[a-h])$",
            r"^(HD[\s\-]?\d+[\s\-]?[a-h])$",
            r"^(GJ[\s\-]?\d+[\s\-]?[a-h])$",
            r"^(Kepler[\s\-]?\d+[\s\-]?[a-h])$",
            r"^(K2[\s\-]?\d+[\s\-]?[a-h])$",
            r"^(TOI[\s\-]?[\d\.]+[\s\-]?[a-h])$",
            # Proper names
            r"^(Proxima Centauri [a-h])$",
            r"^(TRAPPIST[\s\-]?\d+[\s\-]?[a-h])$",
        ],
        TargetType.COORDINATES: [
            r"^(\d+\.?\d*)\s*[,\s]\s*([+-]?\d+\.?\d*)$",
        ],
    }

    @classmethod
    def parse(cls, identifier: str) -> tuple[TargetType, str, Optional[str]]:
        """
        Parse identifier and return (type, cleaned_id, planet_letter).

        Returns:
            Tuple of (TargetType, main_id, planet_letter or None)
        """
        identifier = identifier.strip()

        # Check for planet letter suffix
        planet_letter = None
        planet_match = re.search(r"\s+([b-h])$", identifier, re.IGNORECASE)
        if planet_match:
            planet_letter = planet_match.group(1).lower()

        # Try each pattern
        for target_type, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                match = re.match(pattern, identifier, re.IGNORECASE)
                if match:
                    return target_type, match.group(1), planet_letter

        # Default: treat as planet name and try NASA Archive
        return TargetType.PLANET_NAME, identifier, planet_letter

    @classmethod
    def normalize(cls, identifier: str) -> str:
        """Normalize identifier for consistent querying."""
        # Remove extra whitespace
        identifier = " ".join(identifier.split())
        # Standardize separators
        identifier = re.sub(r"[\s\-_]+", " ", identifier)
        return identifier


class UniversalResolver:
    """
    Resolve any identifier to full cross-matched catalog information.

    This is the core engine that makes TransitKit universal.
    """

    def __init__(self, cache_results: bool = True):
        self.cache_results = cache_results
        self._simbad = None
        self._cache: Dict[str, Any] = {}

    def _init_simbad(self):
        """Initialize SIMBAD with extended votable fields."""
        if self._simbad is None:
            Simbad, _, _, _ = _import_astroquery()
            self._simbad = Simbad()
            self._simbad.add_votable_fields(
                "ids", "flux(V)", "flux(K)", "sp", "plx", "distance", "fe_h", "otype"
            )
        return self._simbad

    def resolve(self, identifier: str) -> Dict[str, Any]:
        """
        Master resolution method - resolves ANY identifier.

        Args:
            identifier: Any planet/star identifier

        Returns:
            Dictionary with all resolved information
        """
        if self.cache_results and identifier in self._cache:
            return self._cache[identifier]

        # Parse the identifier
        target_type, parsed_id, planet_letter = IdentifierParser.parse(identifier)

        result = {
            "input": identifier,
            "type": target_type,
            "parsed_id": parsed_id,
            "planet_letter": planet_letter,
            "ids": CrossMatchedIDs(primary_name=identifier),
            "stellar": StellarParameters(),
            "planets": [],
            "available_data": AvailableData(),
            "resolved": False,
            "errors": [],
        }

        try:
            # Route to appropriate resolver
            if target_type == TargetType.TIC:
                self._resolve_from_tic(int(parsed_id), result)
            elif target_type == TargetType.KIC:
                self._resolve_from_kic(int(parsed_id), result)
            elif target_type == TargetType.TOI:
                self._resolve_from_toi(parsed_id, result)
            elif target_type == TargetType.KOI:
                self._resolve_from_koi(parsed_id, result)
            elif target_type == TargetType.EPIC:
                self._resolve_from_epic(int(parsed_id), result)
            elif target_type in (TargetType.HD, TargetType.HIP, TargetType.PLANET_NAME):
                self._resolve_from_name(identifier, result)
            elif target_type == TargetType.COORDINATES:
                coords = parsed_id.split(",")
                self._resolve_from_coords(float(coords[0]), float(coords[1]), result)
            else:
                # Try name-based resolution
                self._resolve_from_name(identifier, result)

            # Always try to get planet parameters
            self._fetch_planet_params(result)

            # Check available data from missions
            self._check_available_data(result)

            result["resolved"] = True

        except Exception as e:
            result["errors"].append(str(e))
            warnings.warn(f"Resolution incomplete: {e}")

        if self.cache_results:
            self._cache[identifier] = result

        return result

    def _resolve_from_tic(self, tic_id: int, result: Dict):
        """Resolve from TESS Input Catalog ID."""
        _, Catalogs, _, _ = _import_astroquery()

        # Query TIC
        tic_data = Catalogs.query_criteria(catalog="TIC", ID=tic_id)

        if tic_data is None or len(tic_data) == 0:
            raise ValueError(f"TIC {tic_id} not found")

        row = tic_data[0]

        # Fill stellar parameters
        result["stellar"].ra = float(row["ra"])
        result["stellar"].dec = float(row["dec"])
        result["stellar"].teff = self._safe_float(row.get("Teff"))
        result["stellar"].logg = self._safe_float(row.get("logg"))
        result["stellar"].radius = self._safe_float(row.get("rad"))
        result["stellar"].mass = self._safe_float(row.get("mass"))
        result["stellar"].tmag = self._safe_float(row.get("Tmag"))
        result["stellar"].vmag = self._safe_float(row.get("Vmag"))
        result["stellar"].distance = self._safe_float(row.get("d"))
        result["stellar"].source = "TIC"

        # Fill IDs
        result["ids"].tic = tic_id
        result["ids"].gaia_dr3 = str(row.get("GAIA", "")) or None
        result["ids"].two_mass = str(row.get("TWOMASS", "")) or None
        result["ids"].kic = self._safe_int(row.get("KIC"))

        # Get more IDs from SIMBAD
        self._enrich_from_simbad(result)

    def _resolve_from_kic(self, kic_id: int, result: Dict):
        """Resolve from Kepler Input Catalog ID."""
        _, Catalogs, _, _ = _import_astroquery()

        # Query KIC via MAST
        kic_data = Catalogs.query_criteria(catalog="TIC", KIC=kic_id)

        if kic_data is not None and len(kic_data) > 0:
            # Use TIC data but note KIC origin
            row = kic_data[0]
            result["stellar"].ra = float(row["ra"])
            result["stellar"].dec = float(row["dec"])
            result["stellar"].teff = self._safe_float(row.get("Teff"))
            result["stellar"].logg = self._safe_float(row.get("logg"))
            result["stellar"].radius = self._safe_float(row.get("rad"))
            result["stellar"].mass = self._safe_float(row.get("mass"))
            result["stellar"].kepmag = self._safe_float(row.get("Kpmag"))
            result["stellar"].source = "TIC/KIC"

            result["ids"].kic = kic_id
            result["ids"].tic = self._safe_int(row.get("ID"))

        self._enrich_from_simbad(result)

    def _resolve_from_toi(self, toi_id: str, result: Dict):
        """Resolve from TESS Object of Interest ID."""
        _, _, _, NasaExoplanetArchive = _import_astroquery()

        # Query ExoFOP-TESS via NASA Archive
        toi_num = toi_id.replace(".", "")

        try:
            # Try NASA Archive TOI table
            toi_data = NasaExoplanetArchive.query_criteria(
                table="toi", where=f"toi LIKE '{toi_id}%'"
            )

            if toi_data is not None and len(toi_data) > 0:
                row = toi_data[0]
                result["ids"].toi = toi_id
                result["ids"].tic = self._safe_int(row.get("tid"))

                # If we got TIC, resolve fully from there
                if result["ids"].tic:
                    self._resolve_from_tic(result["ids"].tic, result)

        except Exception:
            # Fall back to name resolution
            self._resolve_from_name(f"TOI-{toi_id}", result)

    def _resolve_from_koi(self, koi_id: str, result: Dict):
        """Resolve from Kepler Object of Interest ID."""
        _, _, _, NasaExoplanetArchive = _import_astroquery()

        try:
            koi_data = NasaExoplanetArchive.query_criteria(
                table="koi", where=f"kepoi_name LIKE 'K{koi_id}%'"
            )

            if koi_data is not None and len(koi_data) > 0:
                row = koi_data[0]
                result["ids"].koi = koi_id
                result["ids"].kic = self._safe_int(row.get("kepid"))

                if result["ids"].kic:
                    self._resolve_from_kic(result["ids"].kic, result)

        except Exception:
            self._resolve_from_name(f"KOI-{koi_id}", result)

    def _resolve_from_epic(self, epic_id: int, result: Dict):
        """Resolve from K2 EPIC ID."""
        _, Catalogs, _, _ = _import_astroquery()

        # EPIC catalog via MAST
        epic_data = Catalogs.query_criteria(catalog="TIC", EPIC=epic_id)

        if epic_data is not None and len(epic_data) > 0:
            row = epic_data[0]
            result["stellar"].ra = float(row["ra"])
            result["stellar"].dec = float(row["dec"])
            result["stellar"].teff = self._safe_float(row.get("Teff"))
            result["stellar"].source = "TIC/EPIC"

            result["ids"].epic = epic_id
            result["ids"].tic = self._safe_int(row.get("ID"))

        self._enrich_from_simbad(result)

    def _resolve_from_name(self, name: str, result: Dict):
        """Resolve from planet/star name via SIMBAD and NASA Archive."""
        Simbad, Catalogs, _, NasaExoplanetArchive = _import_astroquery()
        SkyCoord, u, _ = _import_astropy()

        # First try NASA Exoplanet Archive
        try:
            # Extract host name (remove planet letter)
            host_name = re.sub(r"\s+[b-h]$", "", name, flags=re.IGNORECASE)

            nea_data = NasaExoplanetArchive.query_criteria(
                table="pscomppars",
                where=f"hostname LIKE '%{host_name}%' OR pl_name LIKE '%{name}%'",
            )

            if nea_data is not None and len(nea_data) > 0:
                row = nea_data[0]
                result["stellar"].ra = self._safe_float(row.get("ra"))
                result["stellar"].dec = self._safe_float(row.get("dec"))
                result["stellar"].teff = self._safe_float(row.get("st_teff"))
                result["stellar"].logg = self._safe_float(row.get("st_logg"))
                result["stellar"].radius = self._safe_float(row.get("st_rad"))
                result["stellar"].mass = self._safe_float(row.get("st_mass"))
                result["stellar"].feh = self._safe_float(row.get("st_met"))
                result["stellar"].distance = self._safe_float(row.get("sy_dist"))
                result["stellar"].spectral_type = str(row.get("st_spectype", "")) or None
                result["stellar"].source = "NASA Exoplanet Archive"

                result["ids"].planet_name = str(row.get("pl_name", ""))
                result["ids"].host_name = str(row.get("hostname", ""))
                result["ids"].tic = self._safe_int(row.get("tic_id"))
                result["ids"].gaia_dr3 = str(row.get("gaia_id", "")) or None

        except Exception as e:
            result["errors"].append(f"NASA Archive query failed: {e}")

        # Then try SIMBAD for additional IDs
        self._enrich_from_simbad(result, name)

        # If we got coordinates, try TIC lookup
        if result["stellar"].ra and result["stellar"].dec:
            try:
                coord = SkyCoord(ra=result["stellar"].ra, dec=result["stellar"].dec, unit="deg")
                tic_data = Catalogs.query_region(coord, radius=0.01 * u.deg, catalog="TIC")
                if tic_data is not None and len(tic_data) > 0:
                    result["ids"].tic = result["ids"].tic or int(tic_data[0]["ID"])
            except Exception:
                pass

    def _resolve_from_coords(self, ra: float, dec: float, result: Dict):
        """Resolve from coordinates."""
        _, Catalogs, _, _ = _import_astroquery()
        SkyCoord, u, _ = _import_astropy()

        coord = SkyCoord(ra=ra, dec=dec, unit="deg")

        # Search TIC
        tic_data = Catalogs.query_region(coord, radius=0.01 * u.deg, catalog="TIC")

        if tic_data is not None and len(tic_data) > 0:
            # Use closest match
            result["ids"].tic = int(tic_data[0]["ID"])
            self._resolve_from_tic(result["ids"].tic, result)

    def _enrich_from_simbad(self, result: Dict, name: Optional[str] = None):
        """Add additional identifiers from SIMBAD."""
        try:
            simbad = self._init_simbad()

            # Query by name or coordinates
            if name:
                simbad_result = simbad.query_object(name)
            elif result["stellar"].ra and result["stellar"].dec:
                SkyCoord, u, _ = _import_astropy()
                coord = SkyCoord(ra=result["stellar"].ra, dec=result["stellar"].dec, unit="deg")
                simbad_result = simbad.query_region(coord, radius=0.01 * u.deg)
            else:
                return

            if simbad_result is None or len(simbad_result) == 0:
                return

            row = simbad_result[0]

            # Parse all IDs
            if "IDS" in row.colnames:
                ids_str = str(row["IDS"])

                # Extract specific IDs
                hd_match = re.search(r"HD\s*(\d+)", ids_str)
                if hd_match:
                    result["ids"].hd = result["ids"].hd or f"HD {hd_match.group(1)}"

                hip_match = re.search(r"HIP\s*(\d+)", ids_str)
                if hip_match:
                    result["ids"].hip = result["ids"].hip or int(hip_match.group(1))

                gaia_match = re.search(r"Gaia DR3\s*(\d+)", ids_str)
                if gaia_match:
                    result["ids"].gaia_dr3 = result["ids"].gaia_dr3 or gaia_match.group(1)

                twomass_match = re.search(r"2MASS\s*J?([\d\+\-]+)", ids_str)
                if twomass_match:
                    result["ids"].two_mass = result["ids"].two_mass or twomass_match.group(1)

                # Store all aliases
                result["ids"].aliases = [a.strip() for a in ids_str.split("|")]

            # Spectral type
            if "SP_TYPE" in row.colnames:
                result["stellar"].spectral_type = result["stellar"].spectral_type or str(
                    row["SP_TYPE"]
                )

        except Exception as e:
            result["errors"].append(f"SIMBAD enrichment failed: {e}")

    def _fetch_planet_params(self, result: Dict):
        """Fetch planetary parameters from NASA Exoplanet Archive."""
        _, _, _, NasaExoplanetArchive = _import_astroquery()

        try:
            # Build query based on available IDs
            host_name = result["ids"].host_name or result["ids"].planet_name
            if not host_name:
                # Try to construct from other IDs
                if result["ids"].tic:
                    host_name = f"TIC {result['ids'].tic}"
                elif result["ids"].kic:
                    host_name = f"KIC {result['ids'].kic}"
                else:
                    return

            # Remove planet letter for host query
            host_name = re.sub(r"\s+[b-h]$", "", host_name, flags=re.IGNORECASE)

            planet_data = NasaExoplanetArchive.query_criteria(
                table="pscomppars", where=f"hostname LIKE '%{host_name}%'"
            )

            if planet_data is None or len(planet_data) == 0:
                return

            for row in planet_data:
                planet = PlanetParameters(
                    name=str(row.get("pl_name", "")),
                    period=self._safe_float(row.get("pl_orbper")),
                    period_err=self._safe_float(row.get("pl_orbpererr1")),
                    t0=self._safe_float(row.get("pl_tranmid")),
                    duration=self._safe_float(row.get("pl_trandur")),
                    depth=self._safe_float(row.get("pl_trandep")),
                    rp_rs=self._safe_float(row.get("pl_ratror")),
                    a_rs=self._safe_float(row.get("pl_ratdor")),
                    inc=self._safe_float(row.get("pl_orbincl")),
                    ecc=self._safe_float(row.get("pl_orbeccen")),
                    omega=self._safe_float(row.get("pl_orblper")),
                    radius=self._safe_float(row.get("pl_rade")),
                    mass=self._safe_float(row.get("pl_bmasse")),
                    teq=self._safe_float(row.get("pl_eqt")),
                    insol=self._safe_float(row.get("pl_insol")),
                    discovery_method=str(row.get("discoverymethod", "")) or None,
                    discovery_facility=str(row.get("disc_facility", "")) or None,
                    discovery_year=self._safe_int(row.get("disc_year")),
                    confirmed=True,
                    source="NASA Exoplanet Archive",
                )
                result["planets"].append(planet)

        except Exception as e:
            result["errors"].append(f"Planet parameter fetch failed: {e}")

    def _check_available_data(self, result: Dict):
        """Check what mission data is available."""
        _, _, Observations, _ = _import_astroquery()

        try:
            if not result["stellar"].ra or not result["stellar"].dec:
                return

            SkyCoord, u, _ = _import_astropy()
            coord = SkyCoord(ra=result["stellar"].ra, dec=result["stellar"].dec, unit="deg")

            # Query MAST for all observations
            obs = Observations.query_region(coord, radius=0.01 * u.deg)

            if obs is None or len(obs) == 0:
                return

            for row in obs:
                mission = str(row.get("obs_collection", "")).upper()

                if "TESS" in mission:
                    # Extract sector info
                    obs_id = str(row.get("obs_id", ""))
                    sector_match = re.search(r"s(\d+)", obs_id)
                    if sector_match:
                        sector = int(sector_match.group(1))
                        if sector not in result["available_data"].tess_sectors:
                            result["available_data"].tess_sectors.append(sector)

                    # Check cadence
                    exp_time = self._safe_float(row.get("t_exptime"))
                    if exp_time:
                        if exp_time < 30:
                            if "20s" not in result["available_data"].tess_cadences:
                                result["available_data"].tess_cadences.append("20s")
                        elif exp_time < 200:
                            if "2min" not in result["available_data"].tess_cadences:
                                result["available_data"].tess_cadences.append("2min")
                        else:
                            if "FFI" not in result["available_data"].tess_cadences:
                                result["available_data"].tess_cadences.append("FFI")

                elif "KEPLER" in mission and "K2" not in mission:
                    quarter_match = re.search(r"Q(\d+)", str(row.get("obs_id", "")))
                    if quarter_match:
                        quarter = int(quarter_match.group(1))
                        if quarter not in result["available_data"].kepler_quarters:
                            result["available_data"].kepler_quarters.append(quarter)

                elif "K2" in mission:
                    camp_match = re.search(r"C(\d+)", str(row.get("obs_id", "")))
                    if camp_match:
                        camp = int(camp_match.group(1))
                        if camp not in result["available_data"].k2_campaigns:
                            result["available_data"].k2_campaigns.append(camp)

                elif "JWST" in mission:
                    program = str(row.get("proposal_id", ""))
                    if program and program not in result["available_data"].jwst_programs:
                        result["available_data"].jwst_programs.append(program)

                    instrument = str(row.get("instrument_name", "")).upper()
                    if instrument and instrument not in result["available_data"].jwst_instruments:
                        result["available_data"].jwst_instruments.append(instrument)

                    # Check for spectroscopy
                    dataproduct = str(row.get("dataproduct_type", "")).lower()
                    if "spectrum" in dataproduct or "spectroscop" in dataproduct:
                        result["available_data"].has_transmission_spectrum = True

            # Sort lists
            result["available_data"].tess_sectors.sort()
            result["available_data"].kepler_quarters.sort()
            result["available_data"].k2_campaigns.sort()

        except Exception as e:
            result["errors"].append(f"Available data check failed: {e}")

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert to float."""
        if value is None:
            return None
        try:
            val = float(value)
            if np.isnan(val) or np.isinf(val):
                return None
            return val
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_int(value) -> Optional[int]:
        """Safely convert to int."""
        if value is None:
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None


class UniversalTarget:
    """
    Main interface for TransitKit v3.0 Universal.

    Enter ANY identifier and get complete access to all data and analysis.

    Examples:
        >>> target = UniversalTarget("WASP-39 b")
        >>> target = UniversalTarget("TIC 374829238")
        >>> target = UniversalTarget("TOI-700 d")
        >>> target = UniversalTarget("Kepler-442b")
        >>> target = UniversalTarget("HD 209458 b")

        # Get all info
        >>> print(target.summary())

        # Download all available light curves
        >>> lcs = target.get_all_lightcurves()

        # Run complete analysis
        >>> results = target.analyze()

        # Generate publication
        >>> target.export_publication("my_planet_paper/")
    """

    def __init__(self, identifier: str, verbose: bool = True):
        """
        Initialize UniversalTarget with any identifier.

        Args:
            identifier: Any planet/star identifier (TIC, KIC, TOI, planet name, etc.)
            verbose: Print resolution progress
        """
        self.identifier = identifier
        self.verbose = verbose

        self._resolver = UniversalResolver()
        self._resolved = None
        self._lightcurves = {}
        self._spectra = {}

        # Auto-resolve on init
        if verbose:
            print(f"🔍 Resolving: {identifier}")
        self._resolve()

    def _resolve(self):
        """Run resolution."""
        self._resolved = self._resolver.resolve(self.identifier)

        if self.verbose:
            self._print_resolution_summary()

    def _print_resolution_summary(self):
        """Print a nice summary of what was resolved."""
        r = self._resolved
        ids = r["ids"]
        stellar = r["stellar"]
        avail = r["available_data"]

        print(f"\n{'='*60}")
        print(f"✅ Resolved: {ids.planet_name or ids.primary_name}")
        print(f"{'='*60}")

        # IDs
        print("\n📋 Identifiers:")
        if ids.tic:
            print(f"   TIC: {ids.tic}")
        if ids.kic:
            print(f"   KIC: {ids.kic}")
        if ids.toi:
            print(f"   TOI: {ids.toi}")
        if ids.koi:
            print(f"   KOI: {ids.koi}")
        if ids.hd:
            print(f"   HD: {ids.hd}")
        if ids.gaia_dr3:
            print(f"   Gaia DR3: {ids.gaia_dr3}")

        # Stellar params
        print("\n⭐ Host Star:")
        if stellar.teff:
            print(f"   Teff: {stellar.teff:.0f} K")
        if stellar.radius:
            print(f"   Radius: {stellar.radius:.3f} R☉")
        if stellar.mass:
            print(f"   Mass: {stellar.mass:.3f} M☉")
        if stellar.distance:
            print(f"   Distance: {stellar.distance:.1f} pc")

        # Planets
        if r["planets"]:
            print(f"\n🪐 Planets ({len(r['planets'])} found):")
            for p in r["planets"]:
                print(
                    f"   {p.name}: P={p.period:.4f}d, Rp={p.radius:.2f} R⊕"
                    if p.period and p.radius
                    else f"   {p.name}"
                )

        # Available data
        print("\n📡 Available Data:")
        if avail.tess_sectors:
            print(f"   TESS: Sectors {avail.tess_sectors} ({avail.tess_cadences})")
        if avail.kepler_quarters:
            print(f"   Kepler: Quarters {avail.kepler_quarters}")
        if avail.k2_campaigns:
            print(f"   K2: Campaigns {avail.k2_campaigns}")
        if avail.jwst_programs:
            print(f"   JWST: Programs {avail.jwst_programs} ({avail.jwst_instruments})")
        if avail.has_transmission_spectrum:
            print(f"   🌈 Transmission spectrum available!")

        if r["errors"]:
            print(f"\n⚠️  Warnings: {len(r['errors'])}")

        print(f"{'='*60}\n")

    # Properties for easy access
    @property
    def ids(self) -> CrossMatchedIDs:
        """All cross-matched identifiers."""
        return self._resolved["ids"]

    @property
    def stellar(self) -> StellarParameters:
        """Host star parameters."""
        return self._resolved["stellar"]

    @property
    def planets(self) -> List[PlanetParameters]:
        """List of known planets."""
        return self._resolved["planets"]

    @property
    def available_data(self) -> AvailableData:
        """Summary of available mission data."""
        return self._resolved["available_data"]

    @property
    def tic(self) -> Optional[int]:
        """TIC ID."""
        return self.ids.tic

    @property
    def kic(self) -> Optional[int]:
        """KIC ID."""
        return self.ids.kic

    @property
    def coords(self) -> tuple:
        """(RA, Dec) in degrees."""
        return (self.stellar.ra, self.stellar.dec)

    def summary(self) -> str:
        """Return text summary of target."""
        lines = [
            f"UniversalTarget: {self.identifier}",
            f"Resolved name: {self.ids.planet_name or self.ids.host_name}",
            f"Coordinates: {self.stellar.ra:.6f}, {self.stellar.dec:.6f}",
            f"TIC: {self.ids.tic}, KIC: {self.ids.kic}",
            f"Teff: {self.stellar.teff} K, R*: {self.stellar.radius} Rsun",
            f"Planets: {len(self.planets)}",
            f"TESS sectors: {self.available_data.tess_sectors}",
            f"Kepler quarters: {self.available_data.kepler_quarters}",
            f"JWST programs: {self.available_data.jwst_programs}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Export all resolved data as dictionary."""
        return {
            "identifier": self.identifier,
            "ids": {
                "primary_name": self.ids.primary_name,
                "planet_name": self.ids.planet_name,
                "host_name": self.ids.host_name,
                "tic": self.ids.tic,
                "kic": self.ids.kic,
                "toi": self.ids.toi,
                "koi": self.ids.koi,
                "epic": self.ids.epic,
                "hd": self.ids.hd,
                "hip": self.ids.hip,
                "gaia_dr3": self.ids.gaia_dr3,
                "two_mass": self.ids.two_mass,
            },
            "stellar": {
                "ra": self.stellar.ra,
                "dec": self.stellar.dec,
                "teff": self.stellar.teff,
                "logg": self.stellar.logg,
                "feh": self.stellar.feh,
                "radius": self.stellar.radius,
                "mass": self.stellar.mass,
                "distance": self.stellar.distance,
            },
            "planets": [
                {
                    "name": p.name,
                    "period": p.period,
                    "t0": p.t0,
                    "duration": p.duration,
                    "depth": p.depth,
                    "radius": p.radius,
                    "mass": p.mass,
                }
                for p in self.planets
            ],
            "available_data": {
                "tess_sectors": self.available_data.tess_sectors,
                "kepler_quarters": self.available_data.kepler_quarters,
                "k2_campaigns": self.available_data.k2_campaigns,
                "jwst_programs": self.available_data.jwst_programs,
            },
        }

    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export to JSON."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2)
        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)
        return json_str

    def __repr__(self):
        name = self.ids.planet_name or self.ids.host_name or self.identifier
        return f"UniversalTarget('{name}')"


# Convenience function
def resolve(identifier: str, verbose: bool = True) -> UniversalTarget:
    """
    Quick resolve any identifier.

    Args:
        identifier: Any planet/star identifier
        verbose: Print progress

    Returns:
        UniversalTarget instance
    """
    return UniversalTarget(identifier, verbose=verbose)
