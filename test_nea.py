from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

data = NasaExoplanetArchive.query_criteria(
    table='pscomppars', 
    where="pl_name LIKE '%WASP-39%'"
)

print('Columns:', data.colnames[:30])
print('---')
row = data[0]
print('pl_orbper:', row['pl_orbper'])
print('pl_rade:', row['pl_rade'])
print('st_teff:', row['st_teff'])
print('st_rad:', row['st_rad'])
print('ra:', row['ra'])
print('sy_dist:', row['sy_dist'])