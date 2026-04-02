package flwr.android_client

import androidx.room.*

@Database(entities = [Input::class], version = 1, autoMigrations = [])
abstract class Db : RoomDatabase() {
    abstract fun inputDao(): InputDao
}

@Entity
data class Input(
    @PrimaryKey val id: Int = 1,
    @ColumnInfo val device_id: String,
    @ColumnInfo val ip: String,
    @ColumnInfo val port: String
)

@Dao
interface InputDao {
    @Query("SELECT * FROM input WHERE id = 1")
    suspend fun get(): Input?

    @Upsert
    suspend fun upsertAll(vararg inputs: Input)
}
