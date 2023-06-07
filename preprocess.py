import numpy as np


def merge_COVID_and_region_data(df_COVID, df_country_region):
    df_COVID = df_COVID.groupby(['Country_Region', 'Date']).agg(
        {'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'}).reset_index()

    df_COVID['country'] = df_COVID['Country_Region'].str.lower()

    df_COVID.loc[df_COVID["country"] == "korea, north", "country"] = 'north korea'
    df_COVID.loc[df_COVID["country"] == "korea, south", "country"] = 'south korea'
    df_COVID.loc[df_COVID["country"] == "bahamas", "country"] = 'the bahamas'
    df_COVID.loc[df_COVID["country"] == "us", "country"] = 'united states'
    df_COVID.loc[df_COVID["country"] == "burma", "country"] = 'myanmar (burma)'
    df_COVID.loc[df_COVID["country"] == "cabo verde", "country"] = 'cape verde'
    df_COVID.loc[df_COVID["country"] == "congo (kinshasa)", "country"] = 'democratic republic of the congo'
    df_COVID.loc[df_COVID["country"] == "taiwan*", "country"] = 'taiwan'
    df_COVID.loc[df_COVID["country"] == "cote d'ivoire", "country"] = 'côte d\'ivoire'
    df_COVID.loc[df_COVID["country"] == "micronesia", "country"] = 'federated states of micronesia'
    df_COVID.loc[df_COVID["country"] == "sao tome and principe", "country"] = 'são tomé and príncipe'

    df_COVID.drop(df_COVID[df_COVID["country"] == 'diamond princess'].index, inplace=True)
    df_COVID.drop(df_COVID[df_COVID["country"] == 'winter olympics 2022'].index, inplace=True)
    df_COVID.drop(df_COVID[df_COVID["country"] == 'summer olympics 2020'].index, inplace=True)
    df_COVID.drop(df_COVID[df_COVID["country"] == 'ms zaandam'].index, inplace=True)

    df_COVID_region = df_COVID.merge(df_country_region, on='country', how='left')

    df_COVID_region.loc[df_COVID_region["country"] == "the bahamas", "meso_region"] = 'Caribbean'
    df_COVID_region.loc[df_COVID_region["country"] == "the bahamas", "macro_region"] = 'Americas'

    df_COVID_region.loc[df_COVID_region["country"] == "antarctica", "meso_region"] = 'Antarctica'
    df_COVID_region.loc[df_COVID_region["country"] == "antarctica", "macro_region"] = 'Antarctica'

    df_COVID_region.loc[df_COVID_region["country"] == "kosovo", "meso_region"] = 'Southern Europe'
    df_COVID_region.loc[df_COVID_region["country"] == "kosovo", "macro_region"] = 'Europe'

    df_COVID_region.loc[df_COVID_region["country"] == "central african republic", "meso_region"] = 'Middle Africa'
    df_COVID_region.loc[df_COVID_region["country"] == "central african republic", "macro_region"] = 'Africa'

    df_COVID_region.loc[df_COVID_region["country"] == "chad", "meso_region"] = 'Middle Africa'
    df_COVID_region.loc[df_COVID_region["country"] == "chad", "macro_region"] = 'Africa'

    df_COVID_region.loc[df_COVID_region["country"] == "congo (brazzaville)", "meso_region"] = 'Middle Africa'
    df_COVID_region.loc[df_COVID_region["country"] == "congo (brazzaville)", "macro_region"] = 'Africa'

    df_COVID_region.loc[df_COVID_region["country"] == "comoros", "meso_region"] = 'Eastern Africa'
    df_COVID_region.loc[df_COVID_region["country"] == "comoros", "macro_region"] = 'Africa'

    df_COVID_region.loc[df_COVID_region["country"] == "dominica", "meso_region"] = 'Caribbean'
    df_COVID_region.loc[df_COVID_region["country"] == "dominica", "macro_region"] = 'Americas'

    df_COVID_region.loc[df_COVID_region["country"] == "djibouti", "meso_region"] = 'Eastern Africa'
    df_COVID_region.loc[df_COVID_region["country"] == "djibouti", "macro_region"] = 'Africa'

    df_COVID_region.loc[df_COVID_region["country"] == "equatorial guinea", "meso_region"] = 'Middle Africa'
    df_COVID_region.loc[df_COVID_region["country"] == "equatorial guinea", "macro_region"] = 'Africa'

    df_COVID_region.loc[df_COVID_region["country"] == "gambia", "meso_region"] = 'Western Africa'
    df_COVID_region.loc[df_COVID_region["country"] == "gambia", "macro_region"] = 'Africa'

    df_COVID_region.loc[df_COVID_region["country"] == "grenada", "meso_region"] = 'Caribbean'
    df_COVID_region.loc[df_COVID_region["country"] == "grenada", "macro_region"] = 'Americas'

    df_COVID_region.loc[df_COVID_region["country"] == "guinea-bissau", "meso_region"] = 'Western Africa'
    df_COVID_region.loc[df_COVID_region["country"] == "guinea-bissau", "macro_region"] = 'Africa'

    df_COVID_region.loc[df_COVID_region["country"] == "holy see", "meso_region"] = 'Southern Europe'
    df_COVID_region.loc[df_COVID_region["country"] == "holy see", "macro_region"] = 'Europe'

    df_COVID_region.loc[df_COVID_region["country"] == "kiribati", "meso_region"] = 'Oceania'
    df_COVID_region.loc[df_COVID_region["country"] == "kiribati", "macro_region"] = 'Oceania'

    df_COVID_region.loc[df_COVID_region["country"] == "marshall islands", "meso_region"] = 'Oceania'
    df_COVID_region.loc[df_COVID_region["country"] == "marshall islands", "macro_region"] = 'Oceania'

    df_COVID_region.loc[df_COVID_region["country"] == "nauru", "meso_region"] = 'Oceania'
    df_COVID_region.loc[df_COVID_region["country"] == "nauru", "macro_region"] = 'Oceania'

    df_COVID_region.loc[df_COVID_region["country"] == "niger", "meso_region"] = 'Western Africa'
    df_COVID_region.loc[df_COVID_region["country"] == "niger", "macro_region"] = 'Africa'

    df_COVID_region.loc[df_COVID_region["country"] == "palau", "meso_region"] = 'Oceania'
    df_COVID_region.loc[df_COVID_region["country"] == "palau", "macro_region"] = 'Oceania'

    df_COVID_region.loc[df_COVID_region["country"] == "saint vincent and the grenadines", "meso_region"] = 'Caribbean'
    df_COVID_region.loc[df_COVID_region["country"] == "saint vincent and the grenadines", "macro_region"] = 'Americas'

    df_COVID_region.loc[df_COVID_region["country"] == "samoa", "meso_region"] = 'Oceania'
    df_COVID_region.loc[df_COVID_region["country"] == "samoa", "macro_region"] = 'Oceania'

    df_COVID_region.loc[df_COVID_region["country"] == "tonga", "meso_region"] = 'Oceania'
    df_COVID_region.loc[df_COVID_region["country"] == "tonga", "macro_region"] = 'Oceania'

    df_COVID_region.loc[df_COVID_region["country"] == "tuvalu", "meso_region"] = 'Oceania'
    df_COVID_region.loc[df_COVID_region["country"] == "tuvalu", "macro_region"] = 'Oceania'

    df_COVID_region.loc[df_COVID_region["country"] == "west bank and gaza", "meso_region"] = 'Western Asia'
    df_COVID_region.loc[df_COVID_region["country"] == "west bank and gaza", "macro_region"] = 'Asia'

    return df_COVID_region


def df_to_dict(names, df_COVID, column):
    dict_df = {}
    for c in names:
        df_current = df_COVID[df_COVID[column] == c].groupby([column, 'Date']).agg(
            {'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'}).reset_index()

        df_current = df_current.sort_values([column, 'Date'], ascending=[True, True])
        df_current['Confirmed'] = df_current['Confirmed'] - df_current['Confirmed'].shift()
        df_current['Confirmed'] = np.where(df_current['Confirmed'] <= 0, df_current['Confirmed'].shift(),
                                           df_current['Confirmed'])
        df_current['Deaths_pure'] = df_current['Deaths'] - df_current['Deaths'].shift()
        df_current['Deaths_pure'] = np.where(df_current['Deaths_pure'] <= 0, df_current['Deaths_pure'].shift(),
                                             df_current['Deaths_pure'])
        dict_df[c] = df_current
    return dict_df
